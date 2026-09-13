"""One real cell, all eight actual frozen parent chains; no adoption."""
import json
from dataclasses import replace
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from frozen_random_function_parents import load_parents
from random_function_module import RandomFunctionM4
from freeze_random_function_private_controls import build_manifest


def main():
    fold = 'MATR:b1'; results = HERE.parent / 'results'
    source = results / 'm4_random_function_private_controls_v1'
    frozen = json.loads((source / 'frozen_selections.json').read_text()); assert frozen == build_manifest(results)
    audit = json.loads((source / 'verification.json').read_text())
    assert audit['status'] == 'PASS' and audit['summary_sha256'] == sa.digest(source / 'summary.json')
    choice = next(r for r in frozen['selections'] if r['fold'] == fold)['choices']['function_private']
    parents, budget, origin = load_parents(fold)
    spec = choice['model']; assert sa.digest(Path(spec['path'])) == spec['sha256']
    branch = joblib.load(spec['path']); assert set(branch.source_keys) == budget
    _, _, cells = sa.split(sa.load_cells('MATR'), fold); c = min(cells, key=lambda c:c.id)
    out = results / 'm4_random_function_end_to_end_pilot_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'request.json', dict(fold=fold, cell_id=c.id, origin=origin, source_branch=spec,
        source_frozen_sha256=sa.digest(source / 'frozen_selections.json'), source_audit_sha256=sa.digest(source / 'verification.json'),
        code_hashes={n:sa.digest(HERE / n) for n in ('random_function_module.py', 'frozen_random_function_parents.py', 'verify_random_function_end_to_end_pilot.py')},
        scope='One representative cell in MATR:b1, all8real parent chains; not all-cell release or new performance evaluation.'))
    cached = source / 'folds' / fold.replace(':', '__') / 'predictions' / (c.id + '.npz')
    errors = dict(prediction=0., serialization=0., mask=0., prefix=0.); manifest = []
    with np.load(cached) as z, threadpool_limits(limits=1):
        for group, (parent, mode) in parents.items():
            module = RandomFunctionM4(parent, branch, choice['selected']['gain'], mode)
            path = out / (group + '4.joblib'); joblib.dump(module, path, compress=3)
            loaded = joblib.load(path); p = module.predict(c)
            errors['prediction'] = max(errors['prediction'], float(np.max(abs(p - z[group + '4']))))
            errors['serialization'] = max(errors['serialization'], float(np.max(abs(p - loaded.predict(c)))))
            yy = c.y.copy(); yy[sa.K:] = 999
            errors['mask'] = max(errors['mask'], float(np.max(abs(p - loaded.predict(replace(c, y=yy))))))
            n = min(sa.K + 3, len(c.y))
            errors['prefix'] = max(errors['prefix'], float(np.max(abs(p[:n-sa.K] - loaded.predict(sa.prefix(c, n))))))
            manifest.append(dict(group=group + '4', artifact=path.name, sha256=sa.digest(path)))
            print(group, 'actual frozen chain replay', flush=True)
    assert len(manifest) == 8 and max(errors.values()) < 1e-8, errors
    assert all(sa.digest(Path(r['path'])) == r['sha256'] for r in origin['artifacts'])
    sa.write_json(out / 'verification.json', dict(status='ONE_CELL_END_TO_END_PASS_NOT_ADOPTED', manifest=manifest, errors=errors,
        request_sha256=sa.digest(out / 'request.json'), cached_predictions_sha256=sa.digest(cached),
        limits='Eight runnable draft adapters, one held cell only. No claim of all365cell end-to-end verification or completed release.'))
    print('PASS real parent chains', errors, flush=True)


if __name__ == '__main__': main()
