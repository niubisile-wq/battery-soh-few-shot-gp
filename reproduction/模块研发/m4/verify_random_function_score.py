"""Independent target arithmetic, all48 SOH groups and paired intervals."""
import csv
import itertools
import json
from pathlib import Path
from collections import defaultdict
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE, FROZEN
from audit_random_function import independent_function_predict
from audit_conditional_mixed import independent_predict
from freeze_random_function import build_manifest
from summarize_strict import paired_stats


def mapped(g, f):
    return g + ('_' + f if '4' in g and f != 'function_uniform' else '')


def main():
    results = HERE.parent / 'results'; root = results / 'm4_random_function_screen_v1'
    parent = results / 'm3_waveform_candidate_v1'; old = results / 'm4_support_cv_screen_v1'
    frozen = json.loads((root / 'frozen_selections.json').read_text()); summary = json.loads((root / 'summary.json').read_text())
    assert frozen == build_manifest(results)
    request = json.loads((root / 'score_request.json').read_text())
    assert request['frozen_sha256'] == sa.digest(root / 'frozen_selections.json')
    assert request['code_sha256'] == sa.digest(HERE / 'score_random_function.py')
    assert all(sa.digest(HERE / n) == h for n, h in request['helper_hashes'].items())
    parents = ['B' + ''.join(s) for n in range(4) for s in itertools.combinations('123', n)]
    families = ('function_uniform', 'function_private', 'function_shared', 'slope_uniform', 'matched_slope')
    datasets = ('XJTU', 'MATR', 'Tongji'); metrics = ('mae', 'rmse', 'p95_ae')
    rows = list(csv.DictReader((root / 'cells.csv').open()))
    lookup = {(r['cell_id'], r['group']):r for r in rows}
    assert len(rows) == len(lookup) == 17520
    cells = [c for ds in datasets for c in sa.load_cells(ds)]
    errors = dict(prediction=0., metric=0., control=0.)
    buckets = defaultdict(list); cell_metrics = {}; seen = set()
    with threadpool_limits(limits=1):
        for fold in frozen['selections']:
            models = {}
            for f, c in fold['choices'].items():
                assert sa.digest(Path(c['model']['path'])) == c['model']['sha256']
                models[f] = joblib.load(c['model']['path'])
            _, _, test = sa.split(cells, fold['fold'])
            for c in test:
                tail = Path('folds') / fold['fold'].replace(':', '__') / 'predictions' / (c.id + '.npz')
                with np.load(parent / tail) as z:
                    y = z['y'].copy(); expected = {g:z[g].copy() for g in parents}
                assert np.array_equal(y, c.y[sa.K:])
                for f, model in models.items():
                    visible = sa.inference_view(c)
                    if f.startswith('function_'):
                        shared = independent_function_predict(model, visible, False)
                        private = independent_function_predict(model, visible, True)
                        q = private if f == 'function_private' else shared if f == 'function_shared' else (shared + private) / 2
                    else:
                        q = (independent_predict(model, visible, 'shared', point_mean=True) + independent_predict(model, visible, 'private', point_mean=True)) / 2
                    gain = fold['choices'][f]['selected']['gain']
                    for g in parents: expected[mapped(g + '4', f)] = (1 - gain) * expected[g] + gain * q
                    expected['branch_' + f] = q
                with np.load(root / tail) as z, np.load(old / tail) as control:
                    assert set(z.files) == set(expected) | {'y'} and np.array_equal(z['y'], y)
                    for g in parents:
                        errors['control'] = max(errors['control'], float(np.max(abs(z[mapped(g + '4', 'slope_uniform')] - control[g + '4_uniform']))))
                    for g, p in expected.items():
                        errors['prediction'] = max(errors['prediction'], float(np.max(abs(p - z[g]))))
                        if g.startswith('branch_'): continue
                        e = (np.asarray(p, float) - np.asarray(y, float)) * 100
                        values = np.array([np.mean(abs(e)), np.sqrt(np.mean(e * e)), np.quantile(abs(e), .95)])
                        r = lookup[c.id, g]; assert r['dataset'] == c.dataset and r['domain'] == c.domain
                        errors['metric'] = max(errors['metric'], float(np.max(abs(values - [float(r[k]) for k in metrics]))))
                        buckets[c.dataset, g, c.domain].append(values)
                        cell_metrics[c.id, g] = (c.dataset, c.domain, values); seen.add((c.id, g))
            print(fold['fold'], 'independent random-function posterior replay', flush=True)
    assert seen == set(lookup)
    domains = defaultdict(list)
    for (ds, g, _), values in buckets.items(): domains[ds, g].append(np.mean(values, axis=0))
    table = {key:np.mean(values, axis=0) for key, values in domains.items()}
    assert len(table) == len(summary['table']) == 144
    for r in summary['table']:
        errors['metric'] = max(errors['metric'], float(np.max(abs(table[r['dataset'], r['group']] - [r[k] for k in metrics]))))
    assert max(errors.values()) < 1e-8, errors
    edges = []
    for n in range(4):
        for subset in itertools.combinations('1234', n):
            for add in sorted(set('1234') - set(subset)):
                edges.append(('B' + ''.join(sorted((*subset, add))), 'B' + ''.join(subset)))
    assert len(edges) == 32
    comparisons = []
    for f in families:
        gates = {a + '<' + b:bool(all(np.all(table[ds, mapped(a, f)] < table[ds, mapped(b, f)]) for ds in datasets)) for a, b in edges}
        assert gates == summary['gates'][f]
        comparisons.extend((f, mapped(a, f), mapped(b, f)) for a, b in edges)
    for f in families[1:]:
        comparisons.extend(('primary_control', g, mapped(g, f)) for g in ('B4', 'B1234'))
    pairs = []
    for family, a, b in comparisons:
        for ds in datasets:
            ids = sorted(cid for cid, g in cell_metrics if g == a and cell_metrics[cid, g][0] == ds)
            for i, k in enumerate(metrics):
                delta = [(cell_metrics[cid, a][2][i] - cell_metrics[cid, b][2][i]) / 100 for cid in ids]
                domain_ids = [cell_metrics[cid, a][1] for cid in ids]
                pairs.append(dict(family=family, comparison=a + '-' + b, dataset=ds, metric=k, **paired_stats(delta, domain_ids)))
        print(family, a, b, 'paired intervals', flush=True)
    assert len(pairs) == 1512
    sa.write_json(root / 'verification.json', dict(status='PASS', rows=17520, edges=160, errors=errors, paired_comparisons=pairs,
        summary_sha256=sa.digest(root / 'summary.json'), code_sha256=sa.digest(Path(__file__)),
        helper_hashes={n:sa.digest(HERE / n) for n in ('audit_random_function.py', 'audit_conditional_mixed.py', 'freeze_random_function.py')},
        limits='Independent target arithmetic and equal-domain metrics, not new source training. Paired intervals uncorrected for repeated development selection. Private/shared separately selected; not matched kernel/gain target-isolation controls. Matched slope has exact primary kernel/gain, but both source and target covariance change. No independent generalization claim.'))
    print('PASS17520rows160edges1512pairs', errors, flush=True)


if __name__ == '__main__': main()
