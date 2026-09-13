"""Frozen five-initialization smooth-function screen, original parents untouched."""
import json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE, FROZEN
from freeze_random_function_fit import build_manifest
from random_function_branch import ConditionalFunction
from conditional_mixed import ConditionalMixed
from score import PARENTS, addition_edges
from evaluate import error_metrics, aggregate, dump_csv

FAMILIES = ('init_1.0', 'init_0.5', 'init_0.8', 'init_1.25', 'init_2.0')


def group_name(g, family):
    return g + ('_' + family if '4' in g and family != 'init_1.0' else '')


def branch_prediction(model, family, cell):
    assert family in FAMILIES
    return ConditionalFunction(model, True).predict(cell)


def main():
    results = HERE.parent / 'results'; root = results / 'm4_random_function_fit_screen_v1'
    frozen = json.loads((root / 'frozen_selections.json').read_text())
    assert frozen == build_manifest(results)
    assert not (root / 'score_request.json').exists(), 'Existing scoring attempt must not be overwritten'
    sa.write_json(root / 'score_request.json', dict(frozen_sha256=sa.digest(root / 'frozen_selections.json'),
        code_sha256=sa.digest(Path(__file__)), families=FAMILIES,
        helper_hashes={n:sa.digest(HERE / n) for n in ('random_function_branch.py', 'random_function_gp.py', 'conditional_mixed.py', 'freeze_random_function_fit.py')}))
    parent = results / 'm3_waveform_candidate_v1'; old = results / 'm4_random_function_private_controls_v1'
    cells = [c for ds in ('XJTU', 'MATR', 'Tongji') for c in sa.load_cells(ds)]
    rows = []; boundary = 0.; control_error = 0.
    with threadpool_limits(limits=1):
        for fold in frozen['selections']:
            models = {f:joblib.load(c['model']['path']) for f, c in fold['choices'].items()}
            _, _, test = sa.split(cells, fold['fold'])
            for c in test:
                tail = Path('folds') / fold['fold'].replace(':', '__') / 'predictions' / (c.id + '.npz')
                with np.load(parent / tail) as z:
                    y = z['y'].copy(); pp = {g:z[g].copy() for g in PARENTS}
                assert np.array_equal(y, c.y[sa.K:])
                for family, model in models.items():
                    p = branch_prediction(model, family, sa.inference_view(c))
                    yy = c.y.copy(); yy[sa.K:] = 999; n = min(sa.K + 3, len(yy))
                    boundary = max(boundary, float(np.max(abs(p - branch_prediction(model, family, replace(c, y=yy))))),
                        float(np.max(abs(p[:n-sa.K] - branch_prediction(model, family, sa.prefix(c, n))))))
                    gain = fold['choices'][family]['selected']['gain']
                    for g in PARENTS: pp[group_name(g + '4', family)] = pp[g] + gain * (p - pp[g])
                    pp['branch_' + family] = p
                with np.load(old / tail) as z:
                    assert np.array_equal(y, z['y'])
                    for g in PARENTS:
                        np.testing.assert_array_equal(pp[g], z[g])
                        control_error = max(control_error, float(np.max(abs(pp[group_name(g + '4', 'init_1.0')] - z[g + '4']))))
                path = root / tail; path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, y=y, **pp)
                for g, p in pp.items():
                    if not g.startswith('branch_'):
                        rows.append(dict(dataset=c.dataset, domain=c.domain, cell_id=c.id, group=g, **error_metrics(y, p)))
            print(fold['fold'], 'five-initialization random-function screen', flush=True)
    assert len(rows) == 17520 and max(boundary, control_error) < 1e-8
    table = aggregate(rows, ['dataset', 'group']); lookup = {(r['dataset'], r['group']):r for r in table}
    failures = {f:{a + '<' + b:[dict(dataset=ds, metric=k, delta_pp=lookup[ds, group_name(a, f)][k] - lookup[ds, group_name(b, f)][k])
        for ds in ('XJTU', 'MATR', 'Tongji') for k in ('mae', 'rmse', 'p95_ae')
        if lookup[ds, group_name(a, f)][k] >= lookup[ds, group_name(b, f)][k]] for a, b in addition_edges()} for f in FAMILIES}
    gates = {f:{edge:not failed for edge, failed in rr.items()} for f, rr in failures.items()}
    original = json.loads((parent / 'candidate.json').read_text())
    parent_error = max(abs(lookup[r['dataset'], r['group']][k] - r[k]) for r in original['table'] for k in ('mae', 'rmse', 'p95_ae'))
    assert parent_error < 1e-8
    for folder in (parent, FROZEN):
        manifest = json.loads((folder / 'candidate.json').read_text())
        assert all(sa.digest(folder / r['artifact']) == r['sha256'] for r in manifest['manifest'])
    dump_csv(root / 'cells.csv', rows); dump_csv(root / 'comparison.csv', table)
    sa.write_json(root / 'summary.json', dict(status='SCREEN_COMPLETE_NOT_ADOPTED', primary='init_1.0', rows=len(rows), table=table,
        failures=failures, gates=gates, boundary=boundary, control_replay_error=control_error, parent_metric_error=parent_error,
        limits='Five fixed source initializations, no best-fit selection. No GP refits, original parents unchanged. Independent target replay and uncertainty pending. Repeated development cohorts, not independent confirmation.'))
    print({f:[edge for edge, passed in gg.items() if not passed] for f, gg in gates.items()}, flush=True)


if __name__ == '__main__': main()


