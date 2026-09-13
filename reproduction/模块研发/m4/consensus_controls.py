"""Matched ungated ensemble and full-source residual controls; no new selection."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from score import PARENTS
from evaluate import aggregate, error_metrics, dump_csv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True)
    root = Path(ap.parse_args().root).resolve()
    assert (root / 'summary.json').exists()
    out = root / 'matched_controls'; out.mkdir(exist_ok=False)
    selected = json.loads((root / 'frozen_selections.json').read_text())['selections']
    original = json.loads((HERE.parent / 'results/m4_source_batch_v1/result.json').read_text())
    folders = {r['fold']: Path(r['correction']) for r in original['folds']}
    manifest = []
    for s in selected:
        paths = {g: folders[s['fold']] / 'models' / (g + '__' + s['selected']['key'] + '.joblib') for g in PARENTS}
        manifest.append(dict(fold=s['fold'], gain=s['selected']['gain'], ensemble=s['models'],
                             plain={g: dict(path=str(p), sha256=sa.digest(p)) for g, p in paths.items()}))
    sa.write_json(out / 'protocol.json', dict(controls=manifest,
        comparison='Same selected key and gain: gated ensemble versus identical ungated members versus full-source residual.',
        limits='No new selection. Favors configuration selected for gated candidate; not independent confirmation. Constant selected case has identical gated/ungated outputs.',
        code_sha256=sa.digest(Path(__file__))))
    cells = [c for ds in ('XJTU', 'MATR', 'Tongji') for c in sa.load_cells(ds)]
    rows = []; diagnostics = []; replay = 0.
    with threadpool_limits(limits=1):
        for s in manifest:
            models = {}
            for family in ('ensemble', 'plain'):
                models[family] = {}
                for g, spec in s[family].items():
                    assert sa.digest(Path(spec['path'])) == spec['sha256']
                    models[family][g] = joblib.load(spec['path'])
            _, _, test = sa.split(cells, s['fold'])
            for c in test:
                tail = Path('folds') / s['fold'].replace(':', '__') / 'predictions' / (c.id + '.npz')
                with np.load(root / tail) as z:
                    assert np.array_equal(z['y'], c.y[sa.K:])
                    pp = {}
                    for g in PARENTS:
                        m = models['ensemble'][g]; view = sa.inference_view(c)
                        if m.learner == 'constant':
                            mean = m.predict(view, z[g]); gate = np.ones_like(mean)
                        else:
                            # Reconstruct independently of components() used by candidate.
                            values = np.stack([member.predict(view, z[g]) for member in m.members])
                            mean = np.mean(values, axis=0); magnitude = np.mean(abs(values), axis=0)
                            gate = np.zeros_like(mean); keep = magnitude > 1e-12
                            gate[keep] = np.minimum(1., abs(mean[keep]) / magnitude[keep])
                        predicted = z[g] + s['gain'] * np.clip(mean * gate, -m.clip, m.clip)
                        replay = max(replay, float(np.max(abs(predicted - z[g + '4']))))
                        pp[g + '4_gated'] = predicted
                        pp[g + '4_ungated'] = z[g] + s['gain'] * mean
                        pp[g + '4_plain_matched'] = z[g] + s['gain'] * models['plain'][g].predict(view, z[g])
                        diagnostics.append(dict(dataset=c.dataset, domain=c.domain, cell_id=c.id, group=g,
                                                gain=s['gain'], mean_gate=float(np.mean(gate)),
                                                fraction_attenuated=float(np.mean(gate < 1 - 1e-10))))
                    path = out / tail; path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, y=z['y'], **pp)
                    for g, pred in pp.items():
                        rows.append(dict(dataset=c.dataset, domain=c.domain, cell_id=c.id, group=g,
                                         **error_metrics(z['y'], pred)))
            print(s['fold'], 'matched controls and gate replay complete', flush=True)
    assert len(rows) == 8760 and replay < 1e-8
    table = aggregate(rows, ['dataset', 'group'])
    dump_csv(out / 'cells.csv', rows); dump_csv(out / 'comparison.csv', table)
    dump_csv(out / 'gate_diagnostics.csv', diagnostics)
    sa.write_json(out / 'summary.json', dict(status='CONTROLS_COMPLETE_NOT_ADOPTED', table=table,
        rows=len(rows), gated_prediction_replay_error=replay,
        limits='Independent gate arithmetic replay; controls metrics use shared evaluator. No source refitting, significance or generalization claim.'))


if __name__ == '__main__': main()
