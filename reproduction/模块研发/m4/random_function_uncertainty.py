"""Paired intervals for all three passing exploratory steps and controls."""
import csv
import itertools
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from source_oof import sa, HERE
from summarize_strict import paired_stats


def main():
    root = HERE.parent / 'results/m4_random_function_sensitivity_v1'
    audit = json.loads((root / 'verification.json').read_text()); summary = json.loads((root / 'summary.json').read_text())
    assert audit['status'] == 'PASS' and audit['summary_sha256'] == sa.digest(root / 'summary.json')
    steps = [.1, .25, .5]; datasets = ('XJTU', 'MATR', 'Tongji'); metrics = ('mae', 'rmse', 'p95_ae')
    assert summary['passing_candidates'] == ['function_private__' + str(s) for s in steps]
    with (root / 'cells.csv').open() as f: rows = list(csv.DictReader(f))
    lookup = {(r['cell_id'], r['group']):r for r in rows}; assert len(lookup) == len(rows) == 75920
    table = {(r['dataset'], r['group']):r for r in summary['table']}
    ids = {ds:sorted(r['cell_id'] for r in rows if r['dataset'] == ds and r['group'] == 'B') for ds in datasets}
    edges = [('B' + ''.join(sorted((*sub, add))), 'B' + ''.join(sub)) for n in range(4)
        for sub in itertools.combinations('1234', n) for add in sorted(set('1234') - set(sub))]
    pairs = []; error = 0.
    for step in steps:
        def name(g, family='function_private'): return g + '__' + family + '__' + str(step) if '4' in g else g
        comparisons = [('ablation', name(a), name(b)) for a, b in edges]
        for control in ('function_uniform', 'function_shared', 'slope_uniform', 'matched_slope'):
            comparisons.extend(('selection_control', name(g), name(g, control)) for g in ('B4', 'B1234'))
        for kind, a, b in comparisons:
            for ds in datasets:
                domains = [lookup[cid, a]['domain'] for cid in ids[ds]]
                assert all(lookup[cid, a]['domain'] == lookup[cid, b]['domain'] for cid in ids[ds])
                for metric in metrics:
                    delta = [(float(lookup[cid, a][metric]) - float(lookup[cid, b][metric])) / 100 for cid in ids[ds]]
                    stats = paired_stats(delta, domains)
                    expected = table[ds, a][metric] - table[ds, b][metric]
                    error = max(error, abs(stats['delta_pp'] - expected))
                    pairs.append(dict(step=step, kind=kind, comparison=a + '-' + b, dataset=ds, metric=metric, **stats))
        print(step, 'full16 and all existing controls paired', flush=True)
    assert len(pairs) == 1080 and error < 1e-10
    out = HERE.parent / 'results/m4_random_function_uncertainty_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'summary.json', dict(status='PAIRED_DEVELOPMENT_INTERVALS_COMPLETE', pairs=pairs, aggregate_error=error,
        summary_sha256=sa.digest(root / 'summary.json'), audit_sha256=sa.digest(root / 'verification.json'),
        cells_sha256=sa.digest(root / 'cells.csv'), code_sha256=sa.digest(Path(__file__)), helper_sha256=sa.digest(HERE.parent / 'm2/summarize_strict.py'),
        limits='All three passing exploratory steps reported. 5000 paired domain and within-domain cell bootstrap samples, no repeated-selection/multiple-comparison correction. Existing controls have independently selected fold settings, except previous uniform-matched slope; not exact private-mode isolation. Not independent confirmation.'))
    for r in pairs:
        if r['step'] == .5 and r['comparison'] == 'B1234__function_private__0.5-B123':
            print(r['dataset'], r['metric'], r['delta_pp'], r['ci95_domain_bootstrap_pp'], flush=True)


if __name__ == '__main__': main()
