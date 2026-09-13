"""Eight-group development ablation, validation-only tuning and frozen parents."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'evaluation_v3'))
from evaluate import sa, aggregate, error_metrics, dump_csv
from geometry import GeometryGP

P = json.loads((HERE/'protocol.json').read_text())
FROZEN = HERE.parent/'results/m2_reference_candidate_v3'
SCREEN = HERE.parent/'results/m2_reference_bagged_screen_v1/folds'
GROUPS = {'B':'Base', 'B1':'Base+M1', 'B2':'Base+M2', 'B12':'Base+M1+M2'}


def run_fold(fold, out, conditional=False, progression=False, support_clock=False, waveform=False):
    with threadpool_limits(limits=1):
        config=json.loads((HERE/'conditional_protocol.json').read_text()) if conditional else P
        if progression:config=json.loads((HERE/'progression_protocol.json').read_text())
        if support_clock:config=json.loads((HERE/'support_clock_protocol.json').read_text())
        if waveform:config=json.loads((HERE/'waveform_protocol.json').read_text())
        cells = sa.load_cells(fold.split(':')[0])
        train, val, test = sa.split(cells, fold)
        name = fold.replace(':', '__'); dest = Path(out)/'folds'/name
        dest.mkdir(parents=True, exist_ok=False)
        manifest = json.loads((FROZEN/'candidate.json').read_text())
        entries = {r['group']:r for r in manifest['manifest'] if r['fold']==fold}
        provenance = json.loads((HERE.parent/'results/m2_strict_ablation_v1/folds'/name/'provenance.json').read_text())
        allowed = defaultdict(list)
        for cid, j in provenance['source_keys']: allowed[cid].append(j)
        assert set(allowed) == {c.id for c in train}
        assert all({tuple(k) for k in r['source_keys']} == {tuple(k) for k in provenance['source_keys']} for r in entries.values())
        job = SCREEN/name/'seed_0'
        cached = {}
        for group in ['Base+M2', 'Base+M1+M2']:
            episodes = joblib.load(job/(group+'_selection_episodes.joblib'))
            sa.check_validation(episodes, [c.id for c in val], [c.id for c in test])
            cached[group] = {e['cell_id']:e for e in episodes}
        vp = {}
        for c in val:
            vp[c.id] = {'B':cached['Base+M2'][c.id]['base'], 'B1':cached['Base+M1+M2'][c.id]['base']}
            for short, group in [('B2','Base+M2'), ('B12','Base+M1+M2')]:
                s = entries[group]['selection']
                vp[c.id][short] = cached[group][c.id]['reference_predictions'][s['reference_view']][s['reference_index']]
        models, predictions = {}, {}
        for view in config['views']:
            for kernel in config['kernels']:
                key = view+'__'+kernel
                if waveform:
                    from waveform import WaveformGP
                    from conditional_geometry import ConditionalGeometry
                    parent=WaveformGP(view,kernel).fit([sa.source_view(c,allowed[c.id]) for c in train],allowed)
                    variants={key+'__'+mode:ConditionalGeometry(parent,mode) for mode in config['modes']}
                elif support_clock:
                    from support_clock import SupportClockGP
                    from conditional_geometry import ConditionalGeometry
                    parent=SupportClockGP(view,kernel).fit([sa.source_view(c,allowed[c.id]) for c in train],allowed)
                    variants={key+'__'+mode:ConditionalGeometry(parent,mode) for mode in config['modes']}
                elif progression:
                    from progression import ProgressionGP
                    from conditional_geometry import ConditionalGeometry
                    parent=ProgressionGP(view,kernel).fit([sa.source_view(c,allowed[c.id]) for c in train],allowed)
                    variants={key+'__'+mode:ConditionalGeometry(parent,mode) for mode in config['modes']}
                elif conditional:
                    from conditional_geometry import ConditionalGeometry
                    src=HERE.parent/'results/m3_geometry_screen_v1/folds'/name/(key+'.joblib')
                    parent=joblib.load(src)
                    assert set(parent.source_keys)=={tuple(k) for k in provenance['source_keys']}
                    variants={key+'__'+mode:ConditionalGeometry(parent,mode) for mode in config['modes']}
                else:
                    variants={key:GeometryGP(view, kernel).fit([sa.source_view(c, allowed[c.id]) for c in train], allowed)}
                for variant,model in variants.items():
                    models[variant] = model
                    predictions[variant] = {c.id:model.predict(sa.inference_view(c)) for c in val}
                    joblib.dump(model, dest/(variant+'.joblib'))
                print(fold, key, 'prepared', flush=True)
        choices, trials = {}, []
        for group in GROUPS:
            scores = []
            for key in models:
                for w in config['weights']:
                    rr = [dict(dataset=c.dataset, domain=c.domain, mae=float(np.mean(abs(
                        (1-w)*vp[c.id][group]+w*predictions[key][c.id]-c.y[sa.K:])))) for c in val]
                    loss = sa.macro(rr); scores.append((loss, w, key))
                    trials.append(dict(group=group, key=key, weight=w, validation_mae=loss))
            loss, w, key = min(scores)
            choices[group] = dict(key=key, weight=w, validation_mae=loss)
        if config.get('shared_full'):
            shared=choices['B12']
            choices={g:dict(key=shared['key'],weight=shared['weight'],validation_mae=next(
                t['validation_mae'] for t in trials if t['group']==g and t['key']==shared['key'] and t['weight']==shared['weight'])) for g in GROUPS}
        # Persist choices before opening any outer-test predictions or computing outer metrics.
        sa.write_json(dest/'selection.json', dict(choices=choices, trials=trials,
            validation_cells=[c.id for c in val], source_keys=provenance['source_keys']))
        rows = []; boundary = 0.
        for c in test:
            parent = {}
            for short, group in [('B2','Base+M2'), ('B12','Base+M1+M2')]:
                with np.load(job/group/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'], c.y[sa.K:])
                    parent[short] = z[manifest['variant']].copy()
                    parent['B' if short=='B2' else 'B1'] = z['parent'].copy()
            pred = dict(parent)
            used = {s['key'] for s in choices.values()}
            branch = {}
            for key in used:
                model = models[key]; p = model.predict(sa.inference_view(c)); branch[key] = p
                changed = c.y.copy(); changed[sa.K:] = 123
                n = min(len(c.y), sa.K+3)
                err = max(float(np.max(abs(p-model.predict(replace(c,y=changed))))),
                    float(np.max(abs(p[:n-sa.K]-model.predict(sa.prefix(c,n))))))
                boundary = max(boundary, err); assert err < 1e-8
            for group, s in choices.items():
                w = s['weight']; pred[group+'3'] = (1-w)*parent[group]+w*branch[s['key']]
            path = dest/'predictions'/(c.id+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, y=c.y[sa.K:], **pred)
            for group, p in pred.items():
                rows.append(dict(dataset=c.dataset, domain=c.domain, cell_id=c.id, group=group, **error_metrics(c.y[sa.K:],p)))
        sa.write_json(dest/'result.json', dict(fold=fold, rows=rows, max_boundary_error=boundary, choices=choices))
        return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--workers',type=int,default=4)
    variants=ap.add_mutually_exclusive_group();variants.add_argument('--conditional',action='store_true');variants.add_argument('--progression',action='store_true')
    variants.add_argument('--support-clock',action='store_true')
    variants.add_argument('--waveform',action='store_true')
    args = ap.parse_args(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((FROZEN/'candidate.json').read_text())
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in frozen['manifest'])
    config=json.loads((HERE/'conditional_protocol.json').read_text()) if args.conditional else P
    if args.progression:config=json.loads((HERE/'progression_protocol.json').read_text())
    if args.support_clock:config=json.loads((HERE/'support_clock_protocol.json').read_text())
    if args.waveform:config=json.loads((HERE/'waveform_protocol.json').read_text())
    paths=[HERE/'protocol.json',HERE/'geometry.py',Path(__file__)]
    if args.conditional:paths.extend([HERE/'conditional_protocol.json',HERE/'conditional_geometry.py'])
    if args.progression:paths.extend([HERE/'progression_protocol.json',HERE/'progression.py',HERE/'conditional_geometry.py'])
    if args.support_clock:paths.extend([HERE/'support_clock_protocol.json',HERE/'support_clock.py',HERE/'progression.py',HERE/'conditional_geometry.py'])
    if args.waveform:paths.extend([HERE/'waveform_protocol.json',HERE/'waveform.py',HERE/'conditional_geometry.py'])
    sa.write_json(out/'run_protocol.json',dict(protocol=config, hashes={p.name:sa.digest(p) for p in paths}))
    folds = sorted({r['fold'] for r in frozen['manifest']}); rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in as_completed([pool.submit(run_fold, f, str(out),args.conditional,args.progression,args.support_clock,args.waveform) for f in folds]):
            rows.extend(task.result()); print('Completed cells',len(rows)//8,flush=True)
    table = aggregate(rows,['dataset','group']); lookup={(r['dataset'],r['group']):r for r in table}
    gates = {f'{a}<{b}':all(lookup[ds,a][m]<lookup[ds,b][m] for ds in P['datasets'] for m in ['mae','rmse','p95_ae'])
        for a,b in [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]}
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in frozen['manifest'])
    assert len(rows)==365*8
    dump_csv(out/'cells.csv',rows); dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,cells=365,folds=len(folds)))
    print(json.dumps(gates),flush=True)


if __name__=='__main__': main()
