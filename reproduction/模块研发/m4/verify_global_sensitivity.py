"""Independent replay of every declared scalar combination and equal-domain gate.

Checks cached source predictions, not source-model fitting or independent generalization.
"""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa, HERE


def main():
    results = HERE.parent / 'results'
    root = results / 'm4_global_sensitivity_v1'
    summary = json.loads((root / 'summary.json').read_text())
    manifest = json.loads((root / 'protocol.json').read_text())
    groups = ['B' + ''.join(s) for n in range(4)
              for s in itertools.combinations('123', n)]
    with (root / 'cells.csv').open() as f:
        rows = list(csv.DictReader(f))
    saved = {(r['cell_id'], r['group']): r for r in rows}
    assert len(saved) == len(rows) == 105120
    for family, (directory, _) in manifest['sources'].items():
        assert sa.digest(results / directory / 'summary.json') == manifest['source_summary_hashes'][family]
    buckets = defaultdict(list)
    max_error = 0.
    count = 0

    def check(cell, group, y, prediction):
        nonlocal max_error, count
        error = (np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)) * 100
        absolute = np.abs(error)
        metrics = (np.mean(absolute), np.sqrt(np.mean(error ** 2)), np.quantile(absolute, .95))
        row = saved[cell.id, group]
        assert row['dataset'] == cell.dataset and row['domain'] == cell.domain
        for key, value in zip(('mae', 'rmse', 'p95_ae'), metrics):
            max_error = max(max_error, abs(value - float(row[key])))
        buckets[cell.dataset, group, cell.domain].append(metrics)
        count += 1

    for dataset in ('XJTU', 'MATR', 'Tongji'):
        for cell in sa.load_cells(dataset):
            tail = Path('folds') / (dataset + '__' + cell.domain) / 'predictions' / (cell.id + '.npz')
            with np.load(results / 'm3_waveform_candidate_v1' / tail) as z:
                y = z['y']; parents = {g: z[g] for g in groups}
            assert np.array_equal(y, cell.y[sa.K:])
            for group, pred in parents.items():
                check(cell, group, y, pred)
            for family, (directory, suffix) in manifest['sources'].items():
                with np.load(results / directory / tail) as z:
                    assert np.array_equal(y, z['y'])
                    for group, parent in parents.items():
                        assert np.array_equal(parent, z[group])
                        for step in manifest['protocol']['sensitivity_steps']:
                            # Algebraically equivalent convex interpolation, independently implemented.
                            pred = (1 - step) * parent + step * z[group + '4' + suffix]
                            check(cell, group + '4__' + family + '__' + str(step), y, pred)
        print(dataset, 'all combinations independently recalculated', flush=True)
    assert count == len(saved) and max_error < 1e-8
    domain_means = defaultdict(list)
    for (dataset, group, _), values in buckets.items():
        domain_means[dataset, group].append(np.mean(values, axis=0))
    table = {key: np.mean(values, axis=0) for key, values in domain_means.items()}
    assert len(table) == len(summary['table'])
    for row in summary['table']:
        assert np.max(np.abs(table[row['dataset'], row['group']] -
                             [row[k] for k in ('mae', 'rmse', 'p95_ae')])) < 1e-8
    edges = []
    for n in range(4):
        for subset in itertools.combinations('1234', n):
            for module in set('1234') - set(subset):
                edges.append(('B' + ''.join(sorted((*subset, module))), 'B' + ''.join(subset)))
    assert len(edges) == 32
    passing = []
    incremental = []
    for tag, gates in summary['gates'].items():
        computed = {}
        for child, parent in edges:
            a = child + '__' + tag if '4' in child else child
            b = parent + '__' + tag if '4' in parent else parent
            computed[child + '<' + parent] = bool(all(np.all(table[d, a] < table[d, b])
                                                       for d in ('XJTU', 'MATR', 'Tongji')))
        assert computed == gates
        if all(computed.values()): passing.append(tag)
        if computed['B1234<B123']: incremental.append(tag)
    assert passing == summary['passing_candidates']
    sa.write_json(root / 'verification.json', dict(
        status='PASS', replayed_metric_rows=count, metric_error=float(max_error),
        candidates=len(summary['gates']), edges_per_candidate=32,
        passing_candidates=passing, full_parent_incremental_passes=incremental,
        summary_sha256=sa.digest(root / 'summary.json'),
        code_sha256=sa.digest(Path(__file__)),
        limits='Replays cached source predictions and three metrics, equal-domain aggregation and gates. Does not refit source models, independently audit family selection, or establish generalization/significance.'))
    print('PASS', passing, 'incremental', incremental, flush=True)


if __name__ == '__main__':
    main()
