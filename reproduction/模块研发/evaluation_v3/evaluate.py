"""Frozen development diagnosis and simple alternatives; no external data access."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'm2'))
import strict_ablation as sa


def error_metrics(y, p):
    e = (np.asarray(p) - np.asarray(y)) * 100
    assert len(e) and np.isfinite(e).all()
    a = abs(e)
    n = max(1, int(np.ceil(len(a) * .05)))
    return dict(n=len(e), mae=float(a.mean()), rmse=float(np.sqrt(np.mean(e**2))),
                p95_ae=float(np.quantile(a, .95)), bias=float(e.mean()),
                bias_fraction=float(abs(e.mean()) / max(a.mean(), 1e-15)),
                top5_error_share=float(np.sort(a)[-n:].sum() / max(a.sum(), 1e-15)))


def aggregate(rows, keys):
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(r)
    result = []
    for key, group in sorted(buckets.items()):
        domains = defaultdict(list)
        for r in group:
            domains[r['domain']].append(r)
        result.append(dict(zip(keys, key), cells=len(group), domains=len(domains),
                           points=sum(r['n'] for r in group),
                           **{m: float(np.mean([np.mean([r[m] for r in rr])
                                               for rr in domains.values()]))
                              for m in ['mae', 'rmse', 'p95_ae', 'bias', 'bias_fraction', 'top5_error_share']}))
    return result


def quality_masks(x, k):
    # These arrays have already been interpolated: not raw completeness scores.
    v, current, t = x[:, 0].astype(float), x[:, 1].astype(float), x[:, 2].astype(float)
    duration = t[:, -1] - t[:, 0]
    ratio = duration[k:] / max(float(np.median(duration[:k])), 1e-12)
    cv = current[k:].std(1) / np.maximum(abs(current[k:].mean(1)), 1e-12)
    backtrack = np.mean(np.diff(v[k:], axis=1) < -1e-6, axis=1)
    return {'duration_short': ratio < .5, 'duration_reference_range': (ratio >= .5) & (ratio <= 2),
            'duration_long': ratio > 2, 'current_variable': cv > .1,
            'current_stable': cv <= .1, 'voltage_backtrack': backtrack > .01,
            'voltage_monotone': backtrack <= .01}


def dump_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    frozen = sa.ROOT / protocol['frozen_candidate']
    candidate = json.loads((frozen / 'candidate.json').read_text())
    before = {r['artifact']: sa.digest(frozen / r['artifact']) for r in candidate['manifest']}
    assert all(before[r['artifact']] == r['sha256'] for r in candidate['manifest'])
    screen = sa.ROOT / '模块研发/results/m2_reference_bagged_screen_v1'
    cells = [c for ds in protocol['development_datasets'] for c in sa.load_cells(ds)]
    cell_rows, bin_rows, selections, inputs = [], [], [], []
    folds = sorted({f'{c.dataset}:{c.domain}' for c in cells})
    with threadpool_limits(limits=1):
        for fold in folds:
            tr, val, test = sa.split(cells, fold)
            name = fold.replace(':', '__')
            job = screen / 'folds' / name / 'seed_0'
            pls_job = sa.ROOT / '模块研发/results/m1_v1/screen_v1/jobs' / ('P04_pls_metric__' + name)
            provenance = json.loads((pls_job / 'provenance.json').read_text())
            original = json.loads((sa.ROOT / '模块研发/results/m2_strict_ablation_v1/folds' / name / 'provenance.json').read_text())
            # Historical provenance stores physical cycle numbers; strict stores row indices.
            train_by_id = {c.id: c for c in tr}
            expected_cycles = {(cid, float(train_by_id[cid].cycle[j]))
                               for cid, j in original['source_keys']}
            assert {tuple(k) for k in provenance['source_keys']} == expected_cycles
            assert set(provenance['validation_cells']) == {c.id for c in val}
            assert set(provenance['development_target_cells']) == {c.id for c in test}
            pls = joblib.load(pls_job / 'model.joblib')
            mode = json.loads((pls_job / 'tuning.json').read_text())['chosen']['mode']
            ve = joblib.load(job / 'Base+M1+M2_selection_episodes.joblib')
            ve = {e['cell_id']: e for e in ve}
            assert set(ve) == {c.id for c in val}
            vr = {'M1': [], 'ordinary_PLS': []}
            for c in val:
                e = ve[c.id]
                assert np.array_equal(e['y'], c.y[sa.K:])
                p = pls.predict(sa.inference_view(c), mode)
                for parent, prediction in [('M1', e['base']), ('ordinary_PLS', p)]:
                    for w in protocol['simple_controls']['physical_weights']:
                        vr[parent].append(dict(domain=c.domain, dataset=c.dataset, weight=w,
                            mae=float(np.mean(abs((1-w)*prediction+w*e['physical']-e['y'])))))
            chosen = {}
            for parent, rows in vr.items():
                scores = [(sa.macro([r for r in rows if r['weight'] == w]), w)
                          for w in protocol['simple_controls']['physical_weights']]
                score, w = min(scores)
                chosen[parent] = w
                selections.append(dict(fold=fold, parent=parent, weight=w, validation_mae_pp=score*100,
                                       validation_cells=len(val), source_keys=len(provenance['source_keys'])))
            for c in test:
                path = job / 'Base+M1+M2' / (c.id + '.npz')
                with np.load(path) as z:
                    assert np.array_equal(z['y'], c.y[sa.K:])
                    predictions = {'M1': z['parent'], 'Full_v3': z[candidate['variant']]}
                    physical = z['physical_control']
                inputs.append(dict(path=str(path.relative_to(sa.ROOT)), sha256=sa.digest(path)))
                with np.load(job / 'Base+M2' / (c.id + '.npz')) as z:
                    assert np.array_equal(z['y'], c.y[sa.K:])
                    predictions.update(GPR=z['parent'], GPR_M2=z[candidate['variant']])
                p = pls.predict(sa.inference_view(c), mode)
                oldpath = pls_job / 'predictions' / (Path(c.id).stem + '.npz')
                with np.load(oldpath) as z:
                    assert np.array_equal(z['y'], c.y[sa.K:])
                    assert np.max(abs(p-z['pred'])) < 1e-9
                predictions['ordinary_PLS'] = p
                predictions['physical'] = physical
                for parent in ['M1', 'ordinary_PLS']:
                    predictions[parent+'_half_physical'] = .5*predictions[parent]+.5*physical
                    w = chosen[parent]
                    predictions[parent+'_val_physical'] = (1-w)*predictions[parent]+w*physical
                y = c.y[sa.K:]
                masks = quality_masks(c.x, sa.K)
                masks.update({'SOH_lt80': y < .8, 'SOH_80_90': (y >= .8) & (y < .9),
                              'SOH_90_95': (y >= .9) & (y < .95), 'SOH_ge95': y >= .95})
                for group, p in predictions.items():
                    identity = dict(dataset=c.dataset, domain=c.domain, cell_id=c.id, group=group)
                    cell_rows.append(dict(**identity, **error_metrics(y, p)))
                    if group in ['GPR', 'M1', 'Full_v3']:
                        for label, mask in masks.items():
                            if mask.any():
                                bin_rows.append(dict(**identity, bin=label, **error_metrics(y[mask], p[mask])))
            print(fold, 'diagnosis and validation-only controls done', flush=True)
    table = aggregate(cell_rows, ['dataset', 'group'])
    for r in candidate['table']:
        mapping = {'Base': 'GPR', 'Base+M1': 'M1', 'Base+M2': 'GPR_M2', 'Base+M1+M2': 'Full_v3'}
        actual = next(t for t in table if t['dataset'] == r['dataset'] and t['group'] == mapping[r['group']])
        for m in ['mae', 'rmse', 'p95_ae']:
            assert abs(actual[m]-r[m]*100) < 1e-8
    after = {p: sa.digest(frozen / p) for p in before}
    assert before == after
    assert len(cell_rows) == 365*10
    report = dict(status='COMPLETE', protocol_sha256=sa.digest(HERE/'protocol.json'),
                  code_sha256=sa.digest(Path(__file__)), frozen_artifacts_verified=len(before),
                  cells=365, folds=len(folds), table=table, selections=selections,
                  bins=aggregate(bin_rows, ['dataset', 'group', 'bin']),
                  domains=aggregate(cell_rows, ['dataset', 'domain', 'group']), inputs=inputs,
                  limits=protocol['limits']+[protocol['diagnosis']['quality'],
                      'Bin means use only participating cells/domains; bins are not independent treatment effects.',
                      'Bias fraction is within-cell absolute signed mean / MAE, not proof of removable error.',
                      'No M3 or held-out evaluation performed by this script.'])
    sa.write_json(out/'summary.json', report)
    dump_csv(out/'cells.csv', cell_rows)
    dump_csv(out/'bins_by_cell.csv', bin_rows)
    dump_csv(out/'comparison.csv', table)
    dump_csv(out/'domains.csv', report['domains'])
    lines = ['# 冻结v3误差与简单替代方案审计', '', '开发结果；全部误差单位为SOH百分点。', '',
             '| 数据集 | 方法 | MAE | RMSE | P95 |', '|---|---|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['group']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['p95_ae']:.4f} |" for r in table]
    lines += ['', '## 限制', ''] + ['- '+s for s in report['limits']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
