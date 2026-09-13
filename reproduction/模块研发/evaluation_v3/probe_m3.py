"""Two prospectively bounded probes on frozen v3. No GP or label-regressor fits."""
from collections import defaultdict
from dataclasses import replace
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from evaluate import sa, HERE, aggregate, error_metrics, dump_csv
from summarize_strict import paired_stats


def correction(cell, prediction, direction, strength):
    p = np.asarray(prediction)
    assert len(p) == len(cell.x)-sa.K
    anchor = float(np.mean(cell.y[:sa.K]))
    if direction == 'degradation_gain':
        return p - np.minimum(.05, strength*np.maximum(anchor-p, 0))
    if direction == 'charge_anchor':
        x = cell.x.astype(float)
        q = np.sum(.5*(x[:, 1, 1:]+x[:, 1, :-1])*np.diff(x[:, 2], axis=1), axis=1)/3600
        assert np.isfinite(q).all() and (q > 0).all()
        reference = anchor*q[sa.K:]/np.mean(q[:sa.K])
        return p + strength*np.clip(reference-p, -.05, .05)
    raise ValueError(direction)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    args = ap.parse_args(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((HERE/'m3_protocol.json').read_text())
    frozen = sa.ROOT/'模块研发/results/m2_reference_candidate_v3'
    original = json.loads((frozen/'candidate.json').read_text())
    hashes = {r['artifact']: sa.digest(frozen/r['artifact']) for r in original['manifest']}
    assert all(hashes[r['artifact']] == r['sha256'] for r in original['manifest'])
    cells = [c for ds in ['XJTU', 'MATR', 'Tongji'] for c in sa.load_cells(ds)]
    folds = sorted({f'{c.dataset}:{c.domain}' for c in cells})
    rows, selections = [], []; replay_max = boundary_max = 0.; start = time.process_time()
    with threadpool_limits(limits=1):
        for fold in folds:
            _, val, test = sa.split(cells, fold); name = fold.replace(':', '__')
            adapter = joblib.load(frozen/'folds'/name/'Base+M1+M2/adapter.joblib')
            screen = sa.ROOT/'模块研发/results/m2_reference_bagged_screen_v1/folds'/name/'seed_0'
            cached = {e['cell_id']: e for e in joblib.load(screen/'Base+M1+M2_selection_episodes.joblib')}
            vp = {}
            for c in val:
                vp[c.id] = adapter.predict(c)
                expected = cached[c.id]['reference_predictions'][adapter.selection['reference_view']][adapter.selection['reference_index']]
                assert np.max(abs(vp[c.id]-expected)) < 1e-9
            choices = {}
            for direction, spec in protocol['candidates'].items():
                scores = []
                for strength in spec['grid']:
                    rr = [dict(dataset=c.dataset, domain=c.domain,
                        mae=float(np.mean(abs(correction(c, vp[c.id], direction, strength)-c.y[sa.K:])))) for c in val]
                    scores.append((sa.macro(rr), strength))
                loss, strength = min(scores); choices[direction] = strength
                selections.append(dict(fold=fold, direction=direction, strength=strength, validation_mae_pp=loss*100))
            for c in test:
                p = adapter.predict(c)
                with np.load(screen/'Base+M1+M2'/(c.id+'.npz')) as z:
                    delta = float(np.max(abs(p-z[original['variant']])))
                    replay_max = max(replay_max, delta); assert delta < 1e-9
                changed_y = c.y.copy(); changed_y[sa.K:] = 123.
                masked = replace(c, y=changed_y)
                short = sa.prefix(c, sa.K+min(3,len(p)))
                alt = adapter.predict(masked); prefix = adapter.predict(short)
                pp = {'Full_v3': p}
                for direction, strength in choices.items():
                    pp[direction] = correction(c, p, direction, strength)
                    check = max(float(np.max(abs(correction(masked, alt, direction, strength)-pp[direction]))),
                                float(np.max(abs(correction(short, prefix, direction, strength)-pp[direction][:len(prefix)]))))
                    boundary_max = max(boundary_max, check); assert check < 1e-8
                for group, pred in pp.items():
                    rows.append(dict(dataset=c.dataset, domain=c.domain, cell_id=c.id, group=group,
                                     **error_metrics(c.y[sa.K:], pred)))
            print(fold, 'two bounded probes and inference boundaries verified', flush=True)
            assert time.process_time()-start < 1800, 'Prospective compute budget exceeded'
    table = aggregate(rows, ['dataset', 'group']); domains = aggregate(rows, ['dataset', 'domain', 'group'])
    gates, pairs = {}, []
    for direction in protocol['candidates']:
        base = [r for r in table if r['group']=='Full_v3']; new = [r for r in table if r['group']==direction]
        lookup = {r['dataset']:r for r in base}
        gain = float(np.mean([lookup[r['dataset']]['mae']-r['mae'] for r in new]))
        baseline = float(np.mean([r['mae'] for r in base]))
        dbase = {(r['dataset'],r['domain']):r for r in domains if r['group']=='Full_v3'}
        worst = max(r['mae']-dbase[r['dataset'],r['domain']]['mae'] for r in domains if r['group']==direction)
        checks = dict(macro_absolute=gain >= .05, macro_relative=gain/baseline >= .02,
            each_dataset_mae=all(r['mae'] < lookup[r['dataset']]['mae'] for r in new),
            tail_guard=all(r[m]-lookup[r['dataset']][m] <= .05 for r in new for m in ['rmse','p95_ae']),
            domain_guard=worst <= .2)
        gates[direction] = dict(pass_point_gate=all(checks.values()), checks=checks,
                               macro_gain_pp=gain, macro_relative_gain=gain/baseline, worst_domain_mae_increase_pp=worst)
        for ds in ['XJTU','MATR','Tongji']:
            a = {r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==direction}
            b = {r['cell_id']:r for r in rows if r['dataset']==ds and r['group']=='Full_v3'}
            for metric in ['mae','rmse','p95_ae']:
                pairs.append(dict(dataset=ds, direction=direction, metric=metric,
                    **paired_stats([(a[c][metric]-b[c][metric])/100 for c in a], [a[c]['domain'] for c in a])))
    assert all(sa.digest(frozen/p)==h for p,h in hashes.items())
    report = dict(status='COMPLETE', protocol_sha256=sa.digest(HERE/'m3_protocol.json'),
                  code_sha256=sa.digest(Path(__file__)), cpu_seconds=time.process_time()-start,
                  frozen_artifacts_verified=len(hashes), cells=365, folds=21, table=table,
                  gates=gates, paired_comparisons=pairs, selections=selections,
                  max_replay_error=replay_max, max_query_label_or_prefix_error=boundary_max,
                  limits=['Development screen, not independent confirmation or novelty proof.',
                          'SOH stages used for diagnosis are not correction inputs.',
                          'No post-result retuning; frozen v3 unchanged.'])
    sa.write_json(out/'summary.json', report);dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    print(json.dumps(gates,indent=2),flush=True)


if __name__=='__main__': main()
