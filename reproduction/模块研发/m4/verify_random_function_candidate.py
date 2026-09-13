"""Independent all-cell exported-chain replay, source identity and full16 metrics."""
import csv
import itertools
import json
from pathlib import Path
from collections import defaultdict
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from random_function_acceptance import evidence_gate
from audit_random_function import independent_function_predict


def main():
    results = HERE.parent / 'results'; root = results / 'm4_random_function_candidate_v1'
    candidate = json.loads((root / 'candidate.json').read_text()); req = json.loads((root / 'request.json').read_text())
    evidence = json.loads((root / 'evidence.json').read_text()); assert evidence == evidence_gate(results)
    assert candidate['status'] == 'END_TO_END_EXPORTED_INDEPENDENT_AUDIT_PENDING'
    assert candidate['request_sha256'] == sa.digest(root / 'request.json') and req['evidence_sha256'] == sa.digest(root / 'evidence.json')
    assert all(sa.digest(HERE.parent / n) == h for n, h in req['live_code_hashes'].items())
    assert all(sa.digest(root / n) == h for n, h in req['source_snapshot'].items())
    with (root / 'cells.csv').open() as f: rows = list(csv.DictReader(f))
    lookup = {(r['cell_id'], r['group']):r for r in rows}; assert len(lookup) == len(rows) == 5840
    parents = ['B' + ''.join(c) for n in range(4) for c in itertools.combinations('123', n)]
    manifest = {(r['fold'], r['group']):r for r in candidate['manifest']}; assert len(manifest) == 168
    data = {r['cell_id']:r for r in candidate['data']}; pred_files = {r['cell_id']:r for r in candidate['predictions']}
    assert len(data) == len(pred_files) == 365
    cells = {ds:sa.load_cells(ds) for ds in ('XJTU', 'MATR', 'Tongji')}
    errors = dict(prediction=0., adapter=0., metric=0.); buckets = defaultdict(list); seen = set()
    with threadpool_limits(limits=1):
        for fold in evidence['frozen_choices']:
            f = fold['fold']; choice = fold['choices']['function_private']; spec = choice['model']
            assert sa.digest(Path(spec['path'])) == spec['sha256']; branch = joblib.load(spec['path'])
            entries = {}; adapters = {}
            for g in parents:
                r = manifest[f, g + '4']; path = root / r['artifact']; assert sa.digest(path) == r['sha256']
                assert r['branch_sha256'] == spec['sha256'] and r['branch_path'] == spec['path']
                assert r['effective_gain'] == choice['selected']['gain'] and r['global_step'] == .5
                m = joblib.load(path); assert m.parent is not None and m.effective_gain == r['effective_gain'] and m.parent_mode == r['parent_mode']
                assert set(m.source_branch.source_keys) == {tuple(k) for k in r['source_keys']} == set(branch.source_keys)
                assert len(set(branch.source_keys)) <= 1000
                for attr in ('theta_', 'X_train_', 'point_alpha_'):
                    np.testing.assert_array_equal(getattr(m.source_branch.gp, attr), getattr(branch.gp, attr))
                adapters[g] = m; entries[g] = r
            complete = root / 'folds' / f.replace(':', '__') / 'complete.json'
            stored = next(r for r in candidate['folds'] if r['fold'] == f); assert sa.digest(complete) == stored['complete_sha256']
            _, _, test = sa.split(cells[f.split(':')[0]], f)
            for c in test:
                assert data[c.id]['path'] == c.path and data[c.id]['sha256'] == sa.digest(sa.ROOT / c.path)
                path = root / pred_files[c.id]['path']; assert sa.digest(path) == pred_files[c.id]['sha256']
                visible = sa.inference_view(c); q = independent_function_predict(branch, visible, True)
                with np.load(path) as z:
                    y = z['y']; assert np.array_equal(y, c.y[sa.K:])
                    assert set(z.files) == {'y'} | set(parents) | {g + '4' for g in parents}
                    for g, m in adapters.items():
                        p = m.parent.predict(visible) if m.parent_mode is None else m.parent.predict(visible, m.parent_mode)
                        child = (1 - m.effective_gain) * p + m.effective_gain * q
                        errors['adapter'] = max(errors['adapter'], float(np.max(abs(child - m.predict(c)))))
                        for group, prediction in ((g, p), (g + '4', child)):
                            errors['prediction'] = max(errors['prediction'], float(np.max(abs(prediction - z[group]))))
                            e = (np.asarray(prediction, float) - np.asarray(y, float)) * 100
                            values = np.array([np.mean(abs(e)), np.sqrt(np.mean(e * e)), np.quantile(abs(e), .95)])
                            row = lookup[c.id, group]; assert row['dataset'] == c.dataset and row['domain'] == c.domain
                            errors['metric'] = max(errors['metric'], float(np.max(abs(values - [float(row[k]) for k in ('mae', 'rmse', 'p95_ae')]))))
                            buckets[c.dataset, group, c.domain].append(values); seen.add((c.id, group))
            print(f, 'independent exported chain/metrics audit', flush=True)
    assert seen == set(lookup) and max(errors.values()) < 1e-8, errors
    domains = defaultdict(list)
    for (ds, g, _), values in buckets.items(): domains[ds, g].append(np.mean(values, axis=0))
    table = {key:np.mean(values, axis=0) for key, values in domains.items()}; assert len(table) == len(candidate['table']) == 48
    for r in candidate['table']:
        errors['metric'] = max(errors['metric'], float(np.max(abs(table[r['dataset'], r['group']] - [r[k] for k in ('mae', 'rmse', 'p95_ae')]))))
    edges = [('B' + ''.join(sorted((*sub, add))), 'B' + ''.join(sub)) for n in range(4)
        for sub in itertools.combinations('1234', n) for add in sorted(set('1234') - set(sub))]
    gates = {a + '<' + b:bool(all(np.all(table[ds, a] < table[ds, b]) for ds in cells)) for a, b in edges}
    assert gates == candidate['gates'] and len(gates) == 32 and all(gates.values()) and max(errors.values()) < 1e-8
    target = root / 'audit_snapshot' / Path(__file__).name; target.parent.mkdir(exist_ok=True); shutil.copy2(Path(__file__), target)
    sa.write_json(root / 'verification.json', dict(status='EXPORTED_CANDIDATE_AUDIT_PASS', cells=365, rows=5840, adapters=168, edges=32,
        errors=errors, candidate_sha256=sa.digest(root / 'candidate.json'), request_sha256=sa.digest(root / 'request.json'),
        code_sha256=sa.digest(Path(__file__)), audit_snapshot_sha256=sa.digest(target),
        helper_hashes={n:sa.digest(HERE / n) for n in ('audit_random_function.py', 'audit_conditional_mixed.py', 'random_function_acceptance.py')},
        limits=evidence['limits'] + ['Export boundary probes cover one cell/fold; full branch boundary checks were separately audited on365cells.']))
    print('PASS168adapters365cells5840rows32edges', errors, flush=True)


if __name__ == '__main__': main()
