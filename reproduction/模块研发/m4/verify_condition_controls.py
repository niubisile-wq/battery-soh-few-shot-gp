"""Audit separately selected geometry families: selection, models, metrics, gates."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from score import PARENTS, addition_edges


def main():
    results = HERE.parent / 'results'
    root = results / 'm4_condition_controls_v1'
    summary = json.loads((root / 'summary.json').read_text())
    manifest = json.loads((root / 'frozen_selections.json').read_text())
    batch = json.loads((results / 'm4_condition_batch_v1/result.json').read_text())
    folders = {r['fold']: Path(r['correction']) for r in batch['folds']}
    with (root / 'cells.csv').open() as f:
        rows = list(csv.DictReader(f))
    saved = {(r['cell_id'], r['group']): r for r in rows}
    assert len(saved) == len(rows) == 8760
    cells = [c for ds in ('XJTU', 'MATR', 'Tongji') for c in sa.load_cells(ds)]
    buckets = defaultdict(list)
    error = replay = validation_error = 0.
    checked = 0
    with threadpool_limits(limits=1):
        for s in manifest['selections']:
            folder = folders[s['fold']]
            original = json.loads((folder / 'result.json').read_text())
            audit = json.loads((folder / 'verification.json').read_text())
            assert audit['status'] == 'PASS'
            assert audit['result_sha256'] == sa.digest(folder / 'result.json')
            vp = joblib.load(folder / 'validation_parents.joblib')
            _, validation, test = sa.split(cells, s['fold'])
            models = {}
            for family, spec in s['choices'].items():
                eligible = [t for t in original['trials'] if t['key'].startswith(family + '__')]
                assert spec['choice'] == min(eligible, key=lambda t: (t['validation_mae'], t['gain'], t['key']))
                models[family] = {}
                for g, model_spec in spec['models'].items():
                    path = Path(model_spec['path'])
                    assert path == folder / 'models' / (g + '__' + spec['choice']['key'] + '.joblib')
                    assert sa.digest(path) == model_spec['sha256']
                    models[family][g] = joblib.load(path)
                values = defaultdict(list)
                for c in validation:
                    base = vp[c.id, 'B123']
                    pred = base + spec['choice']['gain'] * models[family]['B123'].predict(sa.inference_view(c), base)
                    values[c.domain].append(float(np.mean(abs(pred - c.y[sa.K:]))))
                loss = np.mean([np.mean(v) for v in values.values()])
                validation_error = max(validation_error, abs(loss - spec['choice']['validation_mae']))
            for c in test:
                tail = Path('folds') / s['fold'].replace(':', '__') / 'predictions' / (c.id + '.npz')
                with np.load(results / 'm3_waveform_candidate_v1' / tail) as old, np.load(root / tail) as z:
                    assert np.array_equal(old['y'], z['y']) and np.array_equal(z['y'], c.y[sa.K:])
                    expected = {g: old[g] for g in PARENTS}
                    for family, mm in models.items():
                        gain = s['choices'][family]['choice']['gain']
                        for g, model in mm.items():
                            expected[g + '4__' + family] = old[g] + gain * model.predict(sa.inference_view(c), old[g])
                    assert set(z.files) == {'y'} | set(expected)
                    for g, pred in expected.items():
                        replay = max(replay, float(np.max(abs(pred - z[g]))))
                        e = (np.asarray(pred, float) - np.asarray(z['y'], float)) * 100
                        metric = np.array([np.mean(abs(e)), np.sqrt(np.mean(e * e)), np.quantile(abs(e), .95)])
                        row = saved[c.id, g]
                        assert row['dataset'] == c.dataset and row['domain'] == c.domain
                        error = max(error, float(np.max(abs(metric - [float(row[k]) for k in ('mae', 'rmse', 'p95_ae')]))))
                        buckets[c.dataset, g, c.domain].append(metric)
                        checked += 1
            print(s['fold'], 'family selection and24 outputs replayed', flush=True)
    assert checked == 8760 and max(error, replay, validation_error) < 1e-8
    domain_means = defaultdict(list)
    for (ds, g, _), values in buckets.items():
        domain_means[ds, g].append(np.mean(values, axis=0))
    table = {key: np.mean(values, axis=0) for key, values in domain_means.items()}
    assert len(table) == len(summary['table'])
    for row in summary['table']:
        assert np.max(abs(table[row['dataset'], row['group']] - [row[k] for k in ('mae', 'rmse', 'p95_ae')])) < 1e-8
    for family, gates in summary['gates'].items():
        tag = lambda g: g + '__' + family if '4' in g else g
        for a, b in addition_edges():
            result = all(np.all(table[ds, tag(a)] < table[ds, tag(b)]) for ds in ('XJTU', 'MATR', 'Tongji'))
            assert gates[a + '<' + b] == result
    sa.write_json(root / 'verification.json', dict(status='PASS', rows=checked,
        prediction_error=replay, metric_error=error, selected_validation_loss_error=validation_error,
        all64_gates_reproduced=True, summary_sha256=sa.digest(root / 'summary.json'),
        code_sha256=sa.digest(Path(__file__)),
        limits='Checks selection arithmetic over saved trials and selected validation predictions. Does not replay every unselected trial prediction or refit geometry models; uses historical refit audit. Not independent generalization.'))
    print('PASS family controls audit', flush=True)


if __name__ == '__main__': main()
