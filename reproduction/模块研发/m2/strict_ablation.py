"""Fold-local, label-budget-matched four-group M2 correction experiment.

Source and target query labels are masked at model boundaries. Outer query
labels are read only after the calibrator and fusion parameters are frozen.
Historical output directories are never overwritten by this runner.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
M1 = HERE.parent / 'm1'
sys.path.insert(0, str(M1))
from data import (ROOT, K, load_cells, split, source_indices, metrics, natural,
                  digest, write_json)  # noqa: E402
from gp import GPModel  # noqa: E402
from meta_residual import prior  # noqa: E402

PROTOCOL_PATH = HERE / 'protocol_strict_ablation_v1.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
SPECS = json.loads((M1 / 'candidates_v1.json').read_text())
GROUPS = ['Base', 'Base+M1', 'Base+M2', 'Base+M1+M2']
JOBS = {
    'Base': ('screen_v1', 'C00_original'),
    'Base+M1': ('screen_v2', 'P07_pls_concat'),
    'Physical_control': ('screen_v1', 'P02_anchor_physics'),
}
CELLS = None


def signature():
    paths = [Path(__file__), PROTOCOL_PATH, HERE/'meta_residual.py',
             M1/'data.py', M1/'gp.py', M1/'features.py', M1/'protocol.json']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def macro(rows, key='mae'):
    grouped = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            grouped[(r['dataset'], r['domain'])].append(float(r[key]))
    return float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None


def source_view(cell, allowed):
    """Any attempt to use a non-budget source label will encounter NaN."""
    allowed = np.asarray(allowed, dtype=int)
    if not set(range(K)) <= set(allowed):
        raise ValueError('First K labels must be in source budget')
    y = np.full_like(cell.y, np.nan)
    y[allowed] = cell.y[allowed]
    return replace(cell, y=y)


def inference_view(cell):
    y = np.full_like(cell.y, np.nan)
    y[:K] = cell.y[:K]
    return replace(cell, y=y)


def prefix(cell, n):
    return replace(cell, x=cell.x[:n], y=cell.y[:n], cycle=cell.cycle[:n])


def partition_source(train, seed):
    grouped = defaultdict(list)
    for c in train:
        grouped[c.domain].append(c)
    rng = np.random.default_rng(seed)
    halves = [[], []]
    offset = 0
    for _, cs in sorted(grouped.items()):
        cs = sorted(cs, key=lambda c: natural(c.id))
        for j, k in enumerate(rng.permutation(len(cs))):
            halves[(j + offset) % 2].append(cs[int(k)])
        offset = (offset + len(cs)) % 2
    ids = [{c.id for c in h} for h in halves]
    if not all(ids) or ids[0] & ids[1] or ids[0] | ids[1] != {c.id for c in train}:
        raise ValueError('Invalid source cross-fit partition')
    return halves


def predict_components(model, cell, mode):
    visible = inference_view(cell)
    base = model.predict(visible, mode)
    support_prior = prior(model, prefix(visible, K))
    residual = float(np.mean(visible.y[:K] - support_prior))
    if not np.isfinite(base).all() or not np.isfinite(residual):
        raise ValueError('Prediction accessed a forbidden label or is nonfinite')
    return base, residual


def check_validation(episodes, validation_ids, test_ids):
    ids = [r['cell_id'] for r in episodes]
    if len(ids) != len(set(ids)) or set(ids) != set(validation_ids) or set(ids) & set(test_ids):
        raise ValueError('Validation records are not restricted to this outer fold')


def score_validation(episodes, predictions):
    return [{'dataset':r['dataset'], 'domain':r['domain'], 'cell_id':r['cell_id'],
             **metrics(r['y'], p)} for r, p in zip(episodes, predictions, strict=True)]


def choose_parameters(episodes, calibrator, validation_ids, test_ids):
    check_validation(episodes, validation_ids, test_ids)
    calibrated = [calibrator.predict(r['base']) for r in episodes]
    correction_trials = []
    for alpha in PROTOCOL['m2']['correction_alpha']:
        for beta in PROTOCOL['m2']['support_beta']:
            ps = [r['base'] + alpha*(cal-r['base']) + beta*r['residual']
                  for r, cal in zip(episodes, calibrated, strict=True)]
            rows = score_validation(episodes, ps)
            correction_trials.append({'alpha':alpha, 'beta':beta,
                                      **{k:macro(rows,k) for k in ['mae','rmse','p95_ae']}})
    rank = lambda z: (z['mae'], z['rmse'], z['p95_ae'])
    corr = min(correction_trials, key=rank)
    a = [r['base'] + corr['alpha']*(cal-r['base']) + corr['beta']*r['residual']
         for r,cal in zip(episodes, calibrated, strict=True)]
    # Safety repair: retain the parent prediction as an explicit candidate.
    # The previous A/physical-only fusion forced M2 to alter every prediction,
    # even when validation evidence favored the frozen parent.  The parent is
    # the Base or Base+M1 prediction in this episode, so this remains a strict
    # validation-only choice and cannot use outer-test labels.
    fusion_trials = []
    grid = [i/20 for i in range(21)]
    for wa in grid:
        for wp in grid:
            if wa + wp > 1.0000001:
                continue
            wb = 1.0 - wa - wp
            ps = [wp*r['base'] + wa*p + wb*r['physical']
                  for p,r in zip(a,episodes,strict=True)]
            rows = score_validation(episodes, ps)
            fusion_trials.append({'weight_a':wa, 'weight_parent':wp,
                                  'weight_physical':wb,
                                  **{k:macro(rows,k) for k in ['mae','rmse','p95_ae']}})
    fusion = min(fusion_trials, key=rank)
    return {'alpha':corr['alpha'], 'beta':corr['beta'], 'weight_a':fusion['weight_a'],
            'weight_parent':fusion['weight_parent'], 'weight_physical':fusion['weight_physical'],
            'selected_inner_metrics':fusion, 'correction_trials':correction_trials,
            'fusion_trials':fusion_trials, 'selection_cell_ids':sorted(validation_ids)}


def combine(base, residual, physical, calibrator, selection):
    a = base + selection['alpha']*(calibrator.predict(base)-base) + selection['beta']*residual
    wp = selection.get('weight_parent', 0.0)
    wh = selection.get('weight_physical', 1-selection['weight_a']-wp)
    return selection['weight_a']*a + wp*base + wh*physical


def fit_calibrator(train, allowed, spec, config, mode, seed):
    visible = [source_view(c, allowed[c.id]) for c in train]
    xs, ys, used_keys, audits = [], [], [], []
    for held in partition_source(visible, seed):
        held_ids = {c.id for c in held}
        fit = [c for c in visible if c.id not in held_ids]
        idx = {c.id:allowed[c.id] for c in fit}
        model = GPModel(spec, config).fit(fit, idx)
        fold_keys = []
        for c in held:
            selected = allowed[c.id][allowed[c.id] >= K]
            if not len(selected):
                continue
            # Inference receives only the supports and budgeted query signals.
            keep = np.r_[np.arange(K), selected]
            small = replace(c, x=c.x[keep], y=c.y[keep], cycle=c.cycle[keep])
            p = model.predict(inference_view(small), mode)
            xs.append(p); ys.append(c.y[selected])
            fold_keys.extend([[c.id, int(j)] for j in selected])
        used_keys.extend(fold_keys)
        audits.append({'fit_cells':[c.id for c in fit], 'held_cells':sorted(held_ids),
                       'fit_keys':[[c.id,int(j)] for c in fit for j in idx[c.id]],
                       'calibration_keys':fold_keys, 'gp_warnings':model.warnings})
    allowed_keys = {(c.id,int(j)) for c in train for j in allowed[c.id]}
    keys = [tuple(k) for k in used_keys]
    expected = {k for k in allowed_keys if k[1] >= K}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Cross-fitting did not cover exactly the permitted calibration labels')
    x, y = np.concatenate(xs), np.concatenate(ys)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Cross-fitting read unbudgeted labels')
    cal = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x,y)
    return cal, {'n':len(y), 'keys':used_keys, 'crossfit':audits}, x, y


def init_worker():
    global CELLS
    CELLS = [c for ds in PROTOCOL['datasets'] for c in load_cells(ds)]


def load_frozen(group, fold, train, val, test, allowed):
    screen, candidate = JOBS[group]
    job = ROOT/'模块研发/results/m1_v1'/screen/'jobs'/f'{candidate}__{fold.replace(":","__")}'
    prov = json.loads((job/'provenance.json').read_text())
    expected_keys = [[c.id,float(c.cycle[j])] for c in train for j in allowed[c.id]]
    if (expected_keys != prov['source_keys'] or
        {c.id for c in train} != set(prov['train_cells']) or
        {c.id for c in val} != set(prov['validation_cells']) or
        {c.id for c in test} != set(prov['development_target_cells'])):
        raise ValueError(f'Frozen data mismatch for {job}')
    current_hashes = {name:digest(M1/name) for name in ['data.py','features.py','gp.py']}
    changed_code = [name for name,value in current_hashes.items() if value != prov['code_sha256'][name]]
    chosen = json.loads((job/'tuning.json').read_text())['chosen']
    old = {r['cell_id']:r for r in json.loads((job/'cell_results.json').read_text())}
    return joblib.load(job/'model.joblib'), chosen, old, {
        'candidate':candidate, 'job':str(job.relative_to(ROOT)),
        'model_sha256':digest(job/'model.joblib'),
        'tuning_sha256':digest(job/'tuning.json'), 'mode':chosen['mode'], 'config':chosen['config'],
        'historical_code_sha256':prov['code_sha256'], 'current_code_sha256':current_hashes,
        'changed_since_historical_fit':changed_code}


def run_fold(item):
    fold, out_root = item
    with threadpool_limits(limits=1):
        return run_fold_limited(fold, Path(out_root))


def run_fold_limited(fold, out_root):
    started = time.monotonic()
    job = out_root/'folds'/fold.replace(':','__')
    sig = signature()
    if (job/'complete.json').exists():
        done = json.loads((job/'complete.json').read_text())
        if done['signature'] != sig:
            raise RuntimeError(f'Stale output signature: {job}')
        return done
    job.mkdir(parents=True, exist_ok=True)
    train,val,test = split(CELLS,fold)
    allowed = source_indices(train)
    val_ids, test_ids = {c.id for c in val}, {c.id for c in test}
    source_keys = [[c.id,int(j)] for c in train for j in allowed[c.id]]
    assert len(source_keys) <= 1000
    models, choices, frozen = {}, {}, {}
    cached = {g:{} for g in JOBS}
    max_reproduction_error = 0.0
    raw_max_error = 0.0
    controls = []
    for group in JOBS:
        model, chosen, historical, record = load_frozen(group,fold,train,val,test,allowed)
        # Later M1 candidate additions changed file hashes. Verify compatibility
        # by refitting the exact original spec/config/source keys and replaying
        # predictions, rather than silently accepting mismatched hashes.
        refit = GPModel(model.spec,chosen['config']).fit(
            [source_view(c,allowed[c.id]) for c in train],allowed)
        refit_error = 0.0
        models[group], choices[group], frozen[group] = model, chosen, record
        for c in val+test:
            p,r = predict_components(model,c,chosen['mode'])
            cached[group][c.id] = (p,r)
            check = refit.predict(inference_view(c),chosen['mode'])
            refit_error = max(refit_error,float(np.max(abs(p-check))))
        if refit_error > 1e-8:
            raise ValueError(f'Frozen training reproduction failed: {fold} {group} {refit_error}')
        record['refit_max_prediction_error'] = refit_error
        for c in test:
            with np.load(ROOT/historical[c.id]['prediction_file']) as z:
                if not np.array_equal(z['y'],c.y[K:]):
                    raise ValueError('Frozen labels differ')
                error = float(np.max(abs(z['pred']-cached[group][c.id][0])))
                raw_max_error = max(raw_max_error,error)
                if error > 1e-8:
                    raise ValueError(f'Frozen {group} mismatch {fold} {c.id}: {error}')
                max_reproduction_error = max(max_reproduction_error,error)
    provenance = {'fold':fold,'source_keys':source_keys,'source_cells':[c.id for c in train],
                  'validation_cells':sorted(val_ids),'test_cells':sorted(test_ids),
                  'source_cycles':[[c.id,float(c.cycle[j])] for c in train for j in allowed[c.id]],
                  'frozen':frozen,'signature':sig,'protocol':PROTOCOL,
                  'data_sha256':{c.id:digest(ROOT/c.path) for c in train+val+test}}
    write_json(job/'provenance.json',provenance)
    # All tuning is performed before outer-query scoring.
    fitted = {}
    for seed in PROTOCOL['seeds']:
        for parent,group in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
            chosen = choices[parent]
            candidate = frozen[parent]['candidate']
            cal, audit, cx, cy = fit_calibrator(train,allowed,SPECS[candidate],chosen['config'],chosen['mode'],seed)
            val_episodes = [{'cell_id':c.id,'dataset':c.dataset,'domain':c.domain,'y':c.y[K:],
                             'base':cached[parent][c.id][0],'residual':cached[parent][c.id][1],
                             'physical':cached['Physical_control'][c.id][0]} for c in val]
            selection = choose_parameters(val_episodes,cal,val_ids,test_ids)
            target = job/f'seed_{seed}'/group
            target.mkdir(parents=True,exist_ok=True)
            write_json(target/'selection.json',selection)
            write_json(target/'crossfit_audit.json',audit)
            joblib.dump({'calibrator':cal,'selection':selection,'parent':parent,'fold':fold,
                         'seed':seed,'frozen_models':frozen},target/'adapter.joblib',compress=3)
            np.savez_compressed(target/'calibration_oof.npz',pred=cx,y=cy)
            fitted[seed,group] = (cal,selection)
    # Scoring begins here. It cannot update fitted calibrators or choices.
    rows = []
    audit_max = {'target_query_label_change':0.,'prefix_change':0.}
    for c in test:
        base = cached['Base'][c.id][0]
        m1 = cached['Base+M1'][c.id][0]
        physical = cached['Physical_control'][c.id][0]
        cfile = job/'controls'/f'{Path(c.id).stem}.npz'
        cfile.parent.mkdir(exist_ok=True)
        np.savez_compressed(cfile,y=c.y[K:],cycle=c.cycle[K:],base=base,m1=m1,physical=physical)
        for group,p in [('Base',base),('Base+M1',m1),('Physical_control',physical)]:
            controls.append({'group':group,'seed':None,'fold':fold,'dataset':c.dataset,'domain':c.domain,
                             'cell_id':c.id,**metrics(c.y[K:],p),'prediction_file':str(cfile.relative_to(ROOT)),
                             'prediction_key':{'Base':'base','Base+M1':'m1','Physical_control':'physical'}[group]})
        for seed in PROTOCOL['seeds']:
            for parent,group in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
                cal,selection = fitted[seed,group]
                b,r = cached[parent][c.id]
                p = combine(b,r,physical,cal,selection)
                path = job/f'seed_{seed}'/group/'predictions'/f'{Path(c.id).stem}.npz'
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,y=c.y[K:],pred=p,cycle=c.cycle[K:])
                rows.append({'group':group,'seed':seed,'fold':fold,'dataset':c.dataset,'domain':c.domain,
                             'cell_id':c.id,**metrics(c.y[K:],p),'prediction_file':str(path.relative_to(ROOT)),
                             'prediction_key':'pred'})
        # An actual battery in each fold verifies inference independence.
        if c.id == test[0].id:
            changed_y = c.y.copy(); changed_y[K:] = 123.0
            changed = replace(c,y=changed_y)
            short = prefix(c,K+min(5,len(c.y)-K))
            for parent,group in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
                cal,selection = fitted[0,group]
                b,r = predict_components(models[parent],changed,choices[parent]['mode'])
                pb,_ = predict_components(models['Physical_control'],changed,choices['Physical_control']['mode'])
                expected = combine(*cached[parent][c.id],physical,cal,selection)
                err = float(np.max(abs(combine(b,r,pb,cal,selection)-expected)))
                audit_max['target_query_label_change'] = max(audit_max['target_query_label_change'],err)
                b,r = predict_components(models[parent],short,choices[parent]['mode'])
                pb,_ = predict_components(models['Physical_control'],short,choices['Physical_control']['mode'])
                err = float(np.max(abs(combine(b,r,pb,cal,selection)-expected[:len(short.y)-K])))
                audit_max['prefix_change'] = max(audit_max['prefix_change'],err)
    if max(audit_max.values()) > 1e-8:
        raise ValueError(f'Inference boundary test failed: {audit_max}')
    write_json(job/'cell_results.json',controls+rows)
    done = {'status':'COMPLETE','fold':fold,'cells':len(test),'seeds':PROTOCOL['seeds'],
            'source_labels':len(source_keys),'max_frozen_reproduction_error':max_reproduction_error,
            'inference_audit':audit_max,'signature':sig,'elapsed_s':time.monotonic()-started}
    write_json(job/'complete.json',done)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',default=str(ROOT/'模块研发/results/m2_strict_ablation_v1'))
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--folds',default='all')
    args = ap.parse_args()
    init_worker()
    folds = sorted({f'{c.dataset}:{c.domain}' for c in CELLS})
    if args.folds != 'all':
        requested = args.folds.split(',')
        if not set(requested) <= set(folds):
            raise ValueError('Unknown outer fold')
        folds = requested
    out = Path(args.out).resolve()
    write_json(out/'request.json',{'status':'REQUESTED','folds':folds,'protocol':PROTOCOL,'signature':signature()})
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        futures = {pool.submit(run_fold,(f,str(out))):f for f in folds}
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({k:v for k,v in result.items() if k!='signature'},ensure_ascii=False),flush=True)
    write_json(out/'request.json',{'status':'COMPLETE','folds':folds,'protocol':PROTOCOL,'signature':signature()})


if __name__ == '__main__':
    main()
