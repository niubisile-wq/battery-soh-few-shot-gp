"""Exact retrospective M3/blend interaction decomposition, no new selection."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from source_oof import sa, HERE
from evaluate import dump_csv


def main():
    results = HERE.parent / 'results'; root = results / 'm4_random_function_screen_v1'
    audit = json.loads((root / 'verification.json').read_text()); summary = json.loads((root / 'summary.json').read_text())
    assert audit['status'] == 'PASS' and audit['summary_sha256'] == sa.digest(root / 'summary.json')
    frozen = json.loads((root / 'frozen_selections.json').read_text())
    with (root / 'cells.csv').open() as f: cells = [r for r in csv.DictReader(f) if r['group'] == 'B123']
    assert len(cells) == 365
    rows = []; identity_error = 0.
    for fold in frozen['selections']:
        ds, domain = fold['fold'].split(':', 1)
        for c in (r for r in cells if r['dataset'] == ds and r['domain'] == domain):
            path = root / 'folds' / fold['fold'].replace(':', '__') / 'predictions' / (c['cell_id'] + '.npz')
            with np.load(path) as z:
                y = np.asarray(z['y'], float); base = np.asarray(z['B12'], float)
                d = np.asarray(z['B123'], float) - base
                before = np.abs(base + d - y) - np.abs(base - y)
                for family, choice in fold['choices'].items():
                    suffix = '' if family == 'function_uniform' else '_' + family
                    gain = choice['selected']['gain']
                    b = np.asarray(z['B124' + suffix], float)
                    a = np.asarray(z['B1234' + suffix], float)
                    identity_error = max(identity_error, float(np.max(abs((a - b) - (1 - gain) * d))))
                    actual = np.abs(a - y) - np.abs(b - y)
                    scaled = (1 - gain) * before
                    rows.append(dict(dataset=ds, domain=domain, cell_id=c['cell_id'], family=family, gain=gain,
                        m3_before_mae_delta_pp=float(np.mean(before) * 100),
                        m3_after_mae_delta_pp=float(np.mean(actual) * 100),
                        scaled_old_mae_delta_pp=float(np.mean(scaled) * 100),
                        interaction_remainder_pp=float(np.mean(actual - scaled) * 100),
                        baseline_error_sign_cross_fraction=float(np.mean((base - y) * (b - y) < 0)),
                        points=len(y)))
    assert len(rows) == 1825 and identity_error < 1e-10
    keys = ('m3_before_mae_delta_pp', 'm3_after_mae_delta_pp', 'scaled_old_mae_delta_pp', 'interaction_remainder_pp', 'baseline_error_sign_cross_fraction')
    buckets = defaultdict(list)
    for r in rows: buckets[r['dataset'], r['domain'], r['family']].append(r)
    folds = [dict(dataset=ds, domain=domain, family=f, **{k:float(np.mean([r[k] for r in rr])) for k in keys}) for (ds, domain, f), rr in buckets.items()]
    lookup = {(r['dataset'], r['group']):r for r in summary['table']}
    aggregate_error = 0.; totals = []
    for ds in ('XJTU', 'MATR', 'Tongji'):
        for family in frozen['selections'][0]['choices']:
            rr = [r for r in folds if r['dataset'] == ds and r['family'] == family]
            total = dict(dataset=ds, family=family, **{k:float(np.mean([r[k] for r in rr])) for k in keys})
            suffix = '' if family == 'function_uniform' else '_' + family
            reported = lookup[ds, 'B1234' + suffix]['mae'] - lookup[ds, 'B124' + suffix]['mae']
            aggregate_error = max(aggregate_error, abs(total['m3_after_mae_delta_pp'] - reported))
            assert abs(total['m3_after_mae_delta_pp'] - total['scaled_old_mae_delta_pp'] - total['interaction_remainder_pp']) < 1e-10
            totals.append(total)
    assert aggregate_error < 1e-10
    out = results / 'm4_random_function_interaction_v1'; out.mkdir(exist_ok=False)
    dump_csv(out / 'cells.csv', rows); dump_csv(out / 'folds.csv', folds)
    sa.write_json(out / 'summary.json', dict(status='RETROSPECTIVE_EXACT_DECOMPOSITION', rows=len(rows), table=totals,
        identity_error=identity_error, aggregate_error=aggregate_error, screen_sha256=sa.digest(root / 'summary.json'),
        audit_sha256=sa.digest(root / 'verification.json'), code_sha256=sa.digest(Path(__file__)),
        limits='Algebraic decomposition, not a causal model or tuning signal. MAE nonlinearity means preserving scaled prediction differences need not preserve error ordering. No new prediction variant or parameter selection.'))
    for r in totals:
        if r['dataset'] == 'XJTU': print(r, flush=True)
    print('PASS1825 cells,105 folds', identity_error, aggregate_error, flush=True)


if __name__ == '__main__': main()
