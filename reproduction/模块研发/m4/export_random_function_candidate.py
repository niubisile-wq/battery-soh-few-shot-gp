"""Export168 actual frozen chains, all365cell end-to-end/serialization replay."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from random_function_acceptance import evidence_gate
from frozen_random_function_parents import load_parents
from random_function_module import RandomFunctionM4
from score import PARENTS, addition_edges
from evaluate import error_metrics, aggregate, dump_csv


def export_fold(root, fold):
    req = json.loads((root / 'request.json').read_text()); evidence = json.loads((root / 'evidence.json').read_text())
    assert req['evidence_sha256'] == sa.digest(root / 'evidence.json')
    assert all(sa.digest(HERE.parent / n) == h for n, h in req['live_code_hashes'].items())
    choice = next(r for r in evidence['frozen_choices'] if r['fold'] == fold)['choices']['function_private']
    parents, budget, origin = load_parents(fold)
    assert origin['m2_manifest_sha256'] == evidence['m2_manifest_sha256'] and origin['m3_manifest_sha256'] == evidence['parent_manifest_sha256']
    spec = choice['model']; assert sa.digest(Path(spec['path'])) == spec['sha256']
    branch = joblib.load(spec['path']); assert set(branch.source_keys) == budget
    dest = root / 'folds' / fold.replace(':', '__'); dest.mkdir(parents=True, exist_ok=False)
    adapters = {}; loaded = {}; manifest = []
    for g, (parent, mode) in parents.items():
        adapter = RandomFunctionM4(parent, branch, choice['selected']['gain'], mode)
        path = dest / (g + '4') / 'adapter.joblib'; path.parent.mkdir(parents=True)
        joblib.dump(adapter, path, compress=3); adapters[g + '4'] = adapter; loaded[g + '4'] = joblib.load(path)
        manifest.append(dict(fold=fold, group=g + '4', artifact=str(path.relative_to(root)), sha256=sa.digest(path),
            source_keys=sorted(budget), effective_gain=adapter.effective_gain, global_step=.5,
            branch_path=spec['path'], branch_sha256=spec['sha256'], parent_mode=mode))
    source = HERE.parent / 'results/m4_random_function_private_controls_v1'
    _, _, test = sa.split(sa.load_cells(fold.split(':')[0]), fold)
    probe = min(test, key=lambda c:c.id).id; rows = []; data = []; files = []
    errors = dict(cached=0., serialization=0., mask=0., prefix=0.)
    with threadpool_limits(limits=1):
        for c in test:
            visible = sa.inference_view(c); pp = {}
            for g, (parent, mode) in parents.items():
                pp[g] = parent.predict(visible) if mode is None else parent.predict(visible, mode)
            for g, adapter in adapters.items():
                p = adapter.predict(c); pp[g] = p
                errors['serialization'] = max(errors['serialization'], float(np.max(abs(p - loaded[g].predict(c)))))
                if c.id == probe:
                    yy = c.y.copy(); yy[sa.K:] = 999; n = min(sa.K + 3, len(yy))
                    errors['mask'] = max(errors['mask'], float(np.max(abs(p - loaded[g].predict(replace(c, y=yy))))))
                    errors['prefix'] = max(errors['prefix'], float(np.max(abs(p[:n-sa.K] - loaded[g].predict(sa.prefix(c, n))))))
            tail = Path('folds') / fold.replace(':', '__') / 'predictions' / (c.id + '.npz')
            with np.load(source / tail) as z:
                y = z['y'].copy(); assert np.array_equal(y, c.y[sa.K:])
                for g, p in pp.items(): errors['cached'] = max(errors['cached'], float(np.max(abs(p - z[g]))))
            assert max(errors.values()) < 1e-8, errors
            path = root / tail; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, y=y, **pp)
            files.append(dict(cell_id=c.id, path=str(path.relative_to(root)), sha256=sa.digest(path)))
            data.append(dict(cell_id=c.id, path=c.path, sha256=sa.digest(sa.ROOT / c.path)))
            for g, p in pp.items(): rows.append(dict(dataset=c.dataset, domain=c.domain, cell_id=c.id, group=g, **error_metrics(y, p)))
    assert len(rows) == 16 * len(test) and len(manifest) == 8
    assert all(sa.digest(Path(r['path'])) == r['sha256'] for r in origin['artifacts'])
    sa.write_json(dest / 'complete.json', dict(status='END_TO_END_EXPORTED_AUDIT_PENDING', fold=fold, rows=rows,
        manifest=manifest, data=data, predictions=files, errors=errors, original_parent_origin=origin, boundary_probe_cell=probe))
    print(fold, len(test), 'all16groups end-to-end export complete', errors, flush=True)


def worker(root, fold):
    log = root / 'logs' / (fold.replace(':', '__') + '.log')
    try:
        with log.open('w') as f:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--fold', fold], stdout=f, stderr=subprocess.STDOUT)
        if p.returncode: return dict(fold=fold, status='FAILED', exit_code=p.returncode, log=str(log))
        path = root / 'folds' / fold.replace(':', '__') / 'complete.json'
        return dict(fold=fold, status='EXPORTED', complete_sha256=sa.digest(path))
    except Exception as exc:
        return dict(fold=fold, status='FAILED', error=repr(exc), log=str(log))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root'); ap.add_argument('--fold'); args = ap.parse_args()
    if args.fold:
        assert args.root; export_fold(Path(args.root), args.fold); return
    assert args.root is None
    results = HERE.parent / 'results'; evidence = evidence_gate(results)
    root = results / 'm4_random_function_candidate_v1'; root.mkdir(exist_ok=False); (root / 'logs').mkdir()
    sa.write_json(root / 'evidence.json', evidence)
    snapshots = {}; live = {}
    for directory in ('m1', 'm2', 'm3', 'm4'):
        for path in sorted((HERE.parent / directory).iterdir()):
            if path.is_file() and path.suffix in ('.py', '.json') and path.name != 'progress.json':
                relative = directory + '/' + path.name; target = root / 'source_snapshot' / relative
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
                live[relative] = sa.digest(path); snapshots['source_snapshot/' + relative] = sa.digest(target)
    folds = sorted(r['fold'] for r in evidence['frozen_choices']); assert len(folds) == len(set(folds)) == 21
    sa.write_json(root / 'request.json', dict(status='FROZEN_EXPORT_REQUEST', folds=folds, workers=3,
        evidence_sha256=sa.digest(root / 'evidence.json'), live_code_hashes=live, source_snapshot=snapshots,
        versions={k:version(k) for k in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')}, python=sys.version,
        limits='All365cell end-to-end and serialization replay, boundary probes one representative cell/fold. No fitting, tuning or adoption.'))
    done = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(worker, root, fold) for fold in folds]):
            r = future.result(); done.append(r); sa.write_json(root / 'progress.json', dict(completed=done, total=21)); print(r, flush=True)
    if any(r['status'] != 'EXPORTED' for r in done):
        sa.write_json(root / 'failed.json', dict(status='EXPORT_HAS_FAILURES', folds=done))
        raise RuntimeError('Export failures retained; inspect failed.json before any retry')
    rows = []; manifest = []; data = []; predictions = []; errors = dict(cached=0., serialization=0., mask=0., prefix=0.)
    for r in done:
        path = root / 'folds' / r['fold'].replace(':', '__') / 'complete.json'; assert sa.digest(path) == r['complete_sha256']
        c = json.loads(path.read_text()); rows.extend(c['rows']); manifest.extend(c['manifest']); data.extend(c['data']); predictions.extend(c['predictions'])
        for key in errors: errors[key] = max(errors[key], c['errors'][key])
    assert len(rows) == 5840 and len(manifest) == 168 and len(data) == len(predictions) == 365
    table = aggregate(rows, ['dataset', 'group']); lookup = {(r['dataset'], r['group']):r for r in table}
    controls = json.loads((results / 'm4_random_function_private_controls_v1/summary.json').read_text())
    table_error = max(abs(r[k] - lookup[r['dataset'], r['group']][k]) for r in controls['table'] if (r['dataset'], r['group']) in lookup for k in ('mae', 'rmse', 'p95_ae'))
    assert table_error < 1e-8
    gates = {a + '<' + b:all(lookup[ds, a][k] < lookup[ds, b][k] for ds in ('XJTU', 'MATR', 'Tongji') for k in ('mae', 'rmse', 'p95_ae')) for a, b in addition_edges()}
    assert len(gates) == 32 and all(gates.values())
    dump_csv(root / 'cells.csv', rows); dump_csv(root / 'comparison.csv', table)
    sa.write_json(root / 'candidate.json', dict(status='END_TO_END_EXPORTED_INDEPENDENT_AUDIT_PENDING', family='function_private', global_step=.5,
        table=table, gates=gates, manifest=manifest, data=data, predictions=predictions, errors=errors, table_error=table_error,
        folds=done, request_sha256=sa.digest(root / 'request.json'), limits=evidence['limits']))
    print('EXPORTED168adapters365cells5840rows; independent audit pending', flush=True)


if __name__ == '__main__': main()
