"""Independent effective-gain replay of all registered sensitivity results."""
import csv
import itertools
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from source_oof import sa, HERE


def main():
    results = HERE.parent / 'results'; root = results / 'm4_random_function_sensitivity_v1'; source = results / 'm4_random_function_screen_v1'
    request = json.loads((root / 'request.json').read_text()); s = json.loads((root / 'summary.json').read_text())
    assert request['source_sha256'] == sa.digest(source / 'summary.json')
    assert request['audit_sha256'] == sa.digest(source / 'verification.json')
    assert request['frozen_sha256'] == sa.digest(source / 'frozen_selections.json')
    assert request['code_sha256'] == sa.digest(HERE / 'random_function_sensitivity.py')
    assert request['protocol_sha256'] == sa.digest(HERE / 'random_function_sensitivity_protocol.json')
    frozen = json.loads((source / 'frozen_selections.json').read_text()); protocol = request['protocol']
    with (root / 'cells.csv').open() as f: rows = list(csv.DictReader(f))
    lookup = {(r['cell_id'], r['group']):r for r in rows}; assert len(rows) == len(lookup) == 75920
    parents = ['B' + ''.join(c) for n in range(4) for c in itertools.combinations('123', n)]
    buckets = defaultdict(list); seen = set(); error = 0.
    byds = {ds:sa.load_cells(ds) for ds in ('XJTU', 'MATR', 'Tongji')}
    for fold in frozen['selections']:
        ds = fold['fold'].split(':')[0]; _, _, cells = sa.split(byds[ds], fold['fold'])
        for c in cells:
            path = source / 'folds' / fold['fold'].replace(':', '__') / 'predictions' / (c.id + '.npz')
            with np.load(path) as z:
                y = np.asarray(z['y'], float); assert np.array_equal(y, c.y[sa.K:])
                predictions = {g:z[g] for g in parents}
                for family in protocol['families']:
                    a = fold['choices'][family]['selected']['gain']; q = z['branch_' + family]
                    for step in protocol['global_steps']:
                        for g in parents:
                            predictions[g + '4__' + family + '__' + str(step)] = (1 - a * step) * z[g] + (a * step) * q
                for g, p in predictions.items():
                    e = 100 * (np.asarray(p, float) - y)
                    values = np.array([np.mean(abs(e)), np.sqrt(np.mean(e ** 2)), np.quantile(abs(e), .95)])
                    r = lookup[c.id, g]; assert r['dataset'] == c.dataset and r['domain'] == c.domain
                    error = max(error, float(np.max(abs(values - [float(r[k]) for k in ('mae', 'rmse', 'p95_ae')]))))
                    buckets[ds, g, c.domain].append(values); seen.add((c.id, g))
        print(fold['fold'], 'independent effective-gain replay', flush=True)
    assert seen == set(lookup)
    groups = defaultdict(list)
    for (ds, g, _), vv in buckets.items(): groups[ds, g].append(np.mean(vv, axis=0))
    table = {key:np.mean(vv, axis=0) for key, vv in groups.items()}; assert len(table) == len(s['table']) == 624
    for r in s['table']:
        error = max(error, float(np.max(abs(table[r['dataset'], r['group']] - [r[k] for k in ('mae', 'rmse', 'p95_ae')]))))
    assert error < 1e-8, error
    edges = [('B' + ''.join(sorted((*sub, add))), 'B' + ''.join(sub)) for n in range(4)
        for sub in itertools.combinations('1234', n) for add in sorted(set('1234') - set(sub))]
    assert len(edges) == 32
    passing = []
    for family in protocol['families']:
        for step in protocol['global_steps']:
            tag = family + '__' + str(step)
            def mapped(g): return g + '__' + tag if '4' in g else g
            gates = {a + '<' + b:bool(all(np.all(table[ds, mapped(a)] < table[ds, mapped(b)]) for ds in byds)) for a, b in edges}
            assert gates == s['gates'][tag]
            if all(gates.values()): passing.append(tag)
    assert passing == s['passing_candidates']
    sa.write_json(root / 'verification.json', dict(status='PASS', rows=len(rows), edges=800, error=error,
        passing_candidates=passing, summary_sha256=sa.digest(root / 'summary.json'), request_sha256=sa.digest(root / 'request.json'),
        code_sha256=sa.digest(Path(__file__)), limits='Independent algebra on audited cached branch predictions. Not new source training or independent confirmation. Stability and paired intervals not certified here.'))
    print('PASS75920rows800edges', passing, error, flush=True)


if __name__ == '__main__': main()
