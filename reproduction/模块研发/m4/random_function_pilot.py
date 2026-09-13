"""Smooth random-function source fit and validation; no outer query scoring."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from mixed_effects_pilot import load_inputs
from random_function_branch import RandomFunctionBranch, ConditionalFunction
from score import PARENTS


def load_control(fold):
    root = HERE.parent / 'results/m4_support_cv_validation_v1'
    batch = json.loads((root / 'result.json').read_text())
    audit = json.loads((root / 'verification.json').read_text())
    request = json.loads((root / 'request.json').read_text())
    assert audit['status'] == 'PASS' and audit['result_sha256'] == sa.digest(root / 'result.json')
    assert audit['request_sha256'] == sa.digest(root / 'request.json')
    assert all(sa.digest(HERE / n) == h for n, h in request['code_hashes'].items())
    entry = next(r for r in batch['selections'] if r['fold'] == fold)
    folder = Path(entry['root'])
    assert sa.digest(folder / 'result.json') == entry['result_sha256']
    r = json.loads((folder / 'result.json').read_text())
    assert sa.digest(folder / 'predictions.joblib') == r['prediction_sha256']
    assert all(sa.digest(Path(m['path'])) == m['sha256'] for m in r['models'])
    return joblib.load(folder / 'predictions.joblib')['predictions'], dict(
        root=str(folder), result_sha256=sa.digest(folder / 'result.json'),
        batch_sha256=sa.digest(root / 'result.json'), audit_sha256=sa.digest(root / 'verification.json'),
        models=r['models'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((HERE / 'random_function_protocol.json').read_text())
    source, val, allowed, vp, refs, origin = load_inputs(args.fold)
    control, control_origin = load_control(args.fold)
    sa.write_json(out / 'request.json', dict(fold=args.fold, protocol=protocol, origin=origin, control=control_origin,
        code_hashes={n:sa.digest(HERE / n) for n in ('random_function_pilot.py', 'random_function_branch.py', 'random_function_gp.py',
            'random_function_protocol.json', 'mixed_effects_pilot.py', 'mixed_effects_gp.py', 'difference_gp.py', 'difference_branch.py', 'conditional_mixed.py', 'multiscale.py')}))
    predictions = {}; rows = []; trials = []; models = []; boundary = 0.
    with threadpool_limits(limits=1):
        for kernel in protocol['kernels']:
            model = RandomFunctionBranch.from_encoder(refs[kernel], source, allowed)
            path = out / ('function__' + kernel + '.joblib'); joblib.dump(model, path, compress=3)
            models.append(dict(kernel=kernel, path=str(path), sha256=sa.digest(path), optimization=model.gp.optimization_, diagnostics=model.gp.diagnostics_))
            for c in val:
                parts = {}
                for family, private in (('function_private', True), ('function_shared', False)):
                    adapter = ConditionalFunction(model, private)
                    p = adapter.predict(sa.inference_view(c)); parts[family] = p
                    yy = c.y.copy(); yy[sa.K:] = 999; n = min(sa.K + 3, len(c.y))
                    boundary = max(boundary, float(np.max(abs(p - adapter.predict(replace(c, y=yy))))),
                        float(np.max(abs(p[:n-sa.K] - adapter.predict(sa.prefix(c, n))))))
                parts['function_uniform'] = .5 * (parts['function_private'] + parts['function_shared'])
                parts['slope_uniform'] = control[c.id, 'uniform__' + kernel]
                for family, p in parts.items():
                    key = family + '__' + kernel; predictions[c.id, key] = p
                    for g in PARENTS:
                        for gain in protocol['gains']:
                            q = vp[c.id, g] + gain * (p - vp[c.id, g])
                            rows.append(dict(key=key, target=family, kernel=kernel, group=g, gain=gain, cell_id=c.id,
                                dataset=c.dataset, domain=c.domain, mae=float(np.mean(abs(q - c.y[sa.K:])))))
            print(args.fold, kernel, model.gp.optimization_, model.gp.diagnostics_, flush=True)
        for family in protocol['families']:
            for kernel in protocol['kernels']:
                key = family + '__' + kernel
                for gain in protocol['gains']:
                    rr = [r for r in rows if r['key'] == key and r['gain'] == gain and r['group'] == 'B123']
                    trials.append(dict(target=family, key=key, gain=gain, validation_mae=sa.macro(rr)))
    assert boundary < 1e-8 and len(trials) == 40
    selected = {f:min([r for r in trials if r['target'] == f], key=lambda r:(r['validation_mae'], r['gain'], r['key'])) for f in protocol['families']}
    joblib.dump(predictions, out / 'predictions.joblib', compress=3)
    sa.write_json(out / 'result.json', dict(status='VALIDATION_COMPLETE_AUDIT_PENDING', selected=selected, rows=rows, trials=trials,
        models=models, boundary=boundary, prediction_sha256=sa.digest(out / 'predictions.joblib')))
    print(selected, flush=True)


if __name__ == '__main__':
    main()
