"""Loss-family validation controls and same-configuration loss swap, all reported."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from score import PARENTS, addition_edges
from evaluate import aggregate, error_metrics, dump_csv


def main():
    results = HERE.parent / 'results'
    batch = json.loads((results / 'm4_robust_batch_v1/result.json').read_text())
    assert batch['status'] == 'VALIDATION_BATCH_COMPLETE' and len(batch['folds']) == 21
    out = results / 'm4_robust_controls_v1'; out.mkdir(exist_ok=False)
    selections = []
    for fold in batch['folds']:
        folder = Path(fold['correction']); r = json.loads((folder / 'result.json').read_text())
        audit = json.loads((folder / 'verification.json').read_text())
        assert audit['status'] == 'PASS' and audit['result_sha256'] == sa.digest(folder / 'result.json')
        choices = {}
        for loss in ('absolute', 'squared'):
            eligible = [t for t in r['trials'] if '__' + loss + '__' in t['key']]
            choices[loss] = min(eligible, key=lambda t: (t['validation_mae'], t['gain'], t['key']))
        chosen = r['selected']; key = chosen['key']
        if key != 'constant':
            view, loss, size = key.split('__')
            key = '__'.join((view, 'squared' if loss == 'absolute' else 'absolute', size))
        choices['loss_swap'] = dict(key=key, gain=chosen['gain'])
        specs = {}
        for label, choice in choices.items():
            paths = {g: folder / 'models' / (g + '__' + choice['key'] + '.joblib') for g in PARENTS}
            specs[label] = dict(choice=choice, models={g: dict(path=str(p), sha256=sa.digest(p)) for g, p in paths.items()})
        selections.append(dict(fold=fold['fold'], controls=specs))
    sa.write_json(out / 'frozen_selections.json', dict(selections=selections,
        protocol='Each loss family independently validation-selected; same-key loss swap uses original mixed selection gain. Shared choices across8parents; report every control without datasetwise mixing.',
        code_sha256=sa.digest(Path(__file__))))
    cells = [c for ds in ('XJTU', 'MATR', 'Tongji') for c in sa.load_cells(ds)]
    rows = []
    with threadpool_limits(limits=1):
        for s in selections:
            _, _, test = sa.split(cells, s['fold']); models = {}
            for label, spec in s['controls'].items():
                models[label] = {}
                for g, p in spec['models'].items():
                    assert sa.digest(Path(p['path'])) == p['sha256']
                    models[label][g] = joblib.load(p['path'])
            for c in test:
                tail = Path('folds') / s['fold'].replace(':', '__') / 'predictions' / (c.id + '.npz')
                with np.load(results / 'm3_waveform_candidate_v1' / tail) as z:
                    y = z['y']; pp = {g: z[g] for g in PARENTS}
                assert np.array_equal(y, c.y[sa.K:])
                for label, mm in models.items():
                    for g, m in mm.items():
                        pp[g + '4__' + label] = pp[g] + s['controls'][label]['choice']['gain'] * m.predict(sa.inference_view(c), pp[g])
                path = out / tail; path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, y=y, **pp)
                for g, pred in pp.items():
                    rows.append(dict(dataset=c.dataset, domain=c.domain, cell_id=c.id, group=g, **error_metrics(y, pred)))
            print(s['fold'], 'all loss controls scored', flush=True)
    assert len(rows) == 11680
    table = aggregate(rows, ['dataset', 'group']); look = {(r['dataset'], r['group']): r for r in table}
    gates = {}
    for label in ('absolute', 'squared', 'loss_swap'):
        tag = lambda g: g + '__' + label if '4' in g else g
        gates[label] = {a + '<' + b: all(look[ds, tag(a)][k] < look[ds, tag(b)][k]
            for ds in ('XJTU', 'MATR', 'Tongji') for k in ('mae', 'rmse', 'p95_ae')) for a, b in addition_edges()}
    dump_csv(out / 'cells.csv', rows); dump_csv(out / 'comparison.csv', table)
    sa.write_json(out / 'summary.json', dict(status='CONTROLS_COMPLETE_NOT_ADOPTED', table=table,
        gates=gates, rows=len(rows), limits='Development controls, not independent generalization or source-refit audit. Loss swap is not reselected.'))


if __name__ == '__main__': main()
