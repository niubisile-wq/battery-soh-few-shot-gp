"""Representative real-cell interface replay; not candidate acceptance."""
import json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from freeze_random_function_private_controls import build_manifest
from random_function_module import RandomFunctionM4
from score import PARENTS


def main():
    results = HERE.parent / 'results'; source = results / 'm4_random_function_private_controls_v1'
    frozen = json.loads((source / 'frozen_selections.json').read_text()); assert frozen == build_manifest(results)
    audit = json.loads((source / 'verification.json').read_text())
    assert audit['status'] == 'PASS' and audit['summary_sha256'] == sa.digest(source / 'summary.json')
    out = results / 'm4_random_function_module_replay_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'request.json', dict(source_frozen_sha256=sa.digest(source / 'frozen_selections.json'),
        source_audit_sha256=sa.digest(source / 'verification.json'), code_sha256=sa.digest(Path(__file__)),
        module_sha256=sa.digest(HERE / 'random_function_module.py'),
        scope='First lexicographic held-domain cell per fold, all8parent arrays. Representative integration only, not new performance or all-cell interface test.'))
    byds = {ds:sa.load_cells(ds) for ds in ('XJTU', 'MATR', 'Tongji')}
    manifest = []; error = 0.; mask = 0.; prefix = 0.; comparisons = 0
    with threadpool_limits(limits=1):
        for fold in frozen['selections']:
            choice = fold['choices']['function_private']; spec = choice['model']
            assert sa.digest(Path(spec['path'])) == spec['sha256']
            model = joblib.load(spec['path'])
            module = RandomFunctionM4(None, model, choice['selected']['gain'])
            path = out / (fold['fold'].replace(':', '__') + '.joblib'); joblib.dump(module, path, compress=3)
            loaded = joblib.load(path)
            _, _, test = sa.split(byds[fold['fold'].split(':')[0]], fold['fold']); c = min(test, key=lambda c:c.id)
            tail = source / 'folds' / fold['fold'].replace(':', '__') / 'predictions' / (c.id + '.npz')
            with np.load(tail) as z:
                assert np.array_equal(c.y[sa.K:], z['y'])
                for g in PARENTS:
                    p = loaded.predict_from_parent(c, z[g]); comparisons += 1
                    error = max(error, float(np.max(abs(p - z[g + '4']))))
                yy = c.y.copy(); yy[sa.K:] = 999; p = loaded.predict_from_parent(c, z['B123'])
                mask = max(mask, float(np.max(abs(p - loaded.predict_from_parent(replace(c, y=yy), z['B123'])))))
                n = min(sa.K + 3, len(c.y))
                prefix = max(prefix, float(np.max(abs(p[:n-sa.K] - loaded.predict_from_parent(sa.prefix(c, n), z['B123'][:n-sa.K])))))
            manifest.append(dict(fold=fold['fold'], cell_id=c.id, artifact=path.name, sha256=sa.digest(path),
                original_branch_sha256=spec['sha256'], effective_gain=loaded.effective_gain, cached_predictions_sha256=sa.digest(tail)))
            print(fold['fold'], 'representative adapter serialization replay', flush=True)
    assert len(manifest) == 21 and comparisons == 168 and max(error, mask, prefix) < 1e-8
    sa.write_json(out / 'verification.json', dict(status='REPRESENTATIVE_INTERFACE_PASS_NOT_ADOPTED',
        cells=21, parent_comparisons=comparisons, error=error, label_mask_error=mask, prefix_error=prefix, manifest=manifest,
        request_sha256=sa.digest(out / 'request.json'),
        limits='Draft modules require externally supplied frozen parent predictions (parent=None). Not complete end-to-end packaged models or adoption. Source-fit sensitivity remains pending.'))
    print('PASS21 representative cells168parent comparisons', error, mask, prefix, flush=True)


if __name__ == '__main__': main()
