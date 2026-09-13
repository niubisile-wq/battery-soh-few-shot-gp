"""Retrospective source-selection transfer; never choose a new configuration."""
import csv
import json
from pathlib import Path
import numpy as np
from source_oof import sa, HERE
from evaluate import dump_csv


def main():
    results = HERE.parent / 'results'
    screen = results / 'm4_source_domain_screen_v1'
    source = results / 'm4_source_domain_selection_v1'
    audit = json.loads((screen / 'verification.json').read_text())
    summary = json.loads((screen / 'summary.json').read_text())
    frozen = json.loads((screen / 'frozen_selections.json').read_text())
    assert audit['status'] == 'PASS'
    assert audit['summary_sha256'] == sa.digest(screen / 'summary.json')
    assert frozen['source_selection_sha256'] == sa.digest(source / 'summary.json')
    selection_audit = json.loads((source / 'verification.json').read_text())
    assert selection_audit['comparison_sha256'] == sa.digest(source / 'comparison.csv')
    with (source / 'comparison.csv').open() as f:
        sr = list(csv.DictReader(f))
    sl = {(r['fold'], r['key'], float(r['gain']), r['group']): r for r in sr}
    with (screen / 'cells.csv').open() as f:
        qr = list(csv.DictReader(f))
    metrics = ('mae', 'rmse', 'p95_ae')
    parents = ('B', 'B1', 'B2', 'B3', 'B12', 'B13', 'B23', 'B123')
    rows = []
    for fold in frozen['selections']:
        ds, domain = fold['fold'].split(':', 1)
        q = [r for r in qr if r['dataset'] == ds and r['domain'] == domain]
        def mean(group, key):
            values = [float(r[key]) for r in q if r['group'] == group]
            assert values
            return float(np.mean(values))
        for family in ('cv', 'uniform'):
            chosen = fold['choices'][family]['selected']
            key, gain = chosen['key'], float(chosen['gain'])
            for parent in parents:
                child = parent + '4' + ('_uniform' if family == 'uniform' else '')
                control = parent + '4_original_' + family
                base = sl[fold['fold'], key, 0., parent]
                selected = sl[fold['fold'], key, gain, parent]
                r = dict(fold=fold['fold'], dataset=ds, domain=domain,
                         family=family, parent=parent, key=key, gain=gain,
                         source_mae_before_pp=100 * float(base['source_validation_mae']),
                         source_mae_delta_pp=100 * (float(selected['source_validation_mae']) - float(base['source_validation_mae'])))
                for metric in metrics:
                    r['query_' + metric + '_before_pp'] = mean(parent, metric)
                    r['query_' + metric + '_delta_pp'] = mean(child, metric) - mean(parent, metric)
                    r['selection_' + metric + '_delta_pp'] = mean(child, metric) - mean(control, metric)
                rows.append(r)
    assert len(rows) == 336
    table = {(r['dataset'], r['group']): r for r in summary['table']}
    error = 0.
    for family in ('cv', 'uniform'):
        for parent in parents:
            child = parent + '4' + ('_uniform' if family == 'uniform' else '')
            for ds in ('XJTU', 'MATR', 'Tongji'):
                rr = [r for r in rows if (r['family'], r['parent'], r['dataset']) == (family, parent, ds)]
                for metric in metrics:
                    delta = np.mean([r['query_' + metric + '_delta_pp'] for r in rr])
                    error = max(error, abs(delta - (table[ds, child][metric] - table[ds, parent][metric])))
    assert error < 1e-10
    focus = [r for r in rows if r['parent'] == 'B123']
    reversals = [r for r in focus if r['source_mae_delta_pp'] < -1e-10 and r['query_mae_delta_pp'] > 1e-10]
    out = results / 'm4_source_domain_transfer_diagnosis_v1'
    out.mkdir(exist_ok=False)
    dump_csv(out / 'folds.csv', rows)
    sa.write_json(out / 'summary.json', dict(
        status='RETROSPECTIVE_DIAGNOSIS_ONLY', rows=len(rows), aggregation_error=error,
        full_parent=focus, source_improves_query_worsens=reversals,
        screen_sha256=sa.digest(screen / 'summary.json'), audit_sha256=sa.digest(screen / 'verification.json'),
        selection_sha256=sa.digest(source / 'summary.json'), code_sha256=sa.digest(Path(__file__)),
        limits='Source scores use sparse original-budget queries; outer scores use full query trajectories. Sign mismatch is not a causal diagnosis. No parameter selection, dataset-specific winner mixing, new training, or independent confirmation.'))
    print('PASS336 rows, aggregate error', error, flush=True)
    for r in focus:
        if r['dataset'] == 'XJTU':
            print(r['family'], r['fold'], 'gain', r['gain'], 'source/query MAE delta', r['source_mae_delta_pp'], r['query_mae_delta_pp'], flush=True)
    print('reversals', [(r['family'], r['fold']) for r in reversals], flush=True)


if __name__ == '__main__':
    main()
