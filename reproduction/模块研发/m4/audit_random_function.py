"""Fresh source refit, independent target difference basis and validation replay."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from mixed_effects_pilot import load_inputs
from random_function_pilot import load_control
from random_function_branch import RandomFunctionBranch, ConditionalFunction
from audit_conditional_mixed import independent_predict
from score import PARENTS


def independent_function_predict(model, cell, private):
    # Alternate target contrast basis. Shared posterior uses point-space mean,
    # avoiding unstable reassociation of source Helmert products.
    z = ConditionalFunction(model).latent(cell); x = model.gp.X_train_; gp = model.gp
    zs = z[:sa.K]; d = np.diff(np.eye(sa.K), axis=0)
    rho = gp.rho_ if private else 0.
    cross = gp.kernel_(zs, x) @ gp.H_.T
    inv_cross = cho_solve(gp.factor_, cross.T)
    ss = gp.kernel_(zs, zs) - cross @ inv_cross + rho * gp.kernel_.k1(zs, zs)
    observed = ss + np.eye(sa.K) * float(gp.kernel_.k2.noise_level)
    mean_s = gp.kernel_(zs, x) @ (gp.H_.T @ gp.alpha_)
    residual = np.asarray(cell.y[:sa.K], float) / gp.y_scale_ - mean_s
    weight = np.linalg.solve(d @ observed @ d.T, d @ residual)
    anchor = np.mean(np.asarray(cell.y[:sa.K], float) / gp.y_scale_ - mean_s - ss @ d.T @ weight)
    pieces = []
    for start in range(sa.K, len(z), 256):
        zz = z[start:start + 256]
        kx = gp.kernel_(zz, x)
        qs = gp.kernel_(zz, zs) - (kx @ gp.H_.T) @ inv_cross + rho * gp.kernel_.k1(zz, zs)
        pieces.append((kx @ (gp.H_.T @ gp.alpha_) + qs @ d.T @ weight + anchor) * gp.y_scale_)
    return np.concatenate(pieces)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True)
    root = Path(ap.parse_args().root).resolve()
    req = json.loads((root / 'request.json').read_text()); result = json.loads((root / 'result.json').read_text())
    assert all(sa.digest(HERE / n) == h for n, h in req['code_hashes'].items())
    assert sa.digest(root / 'predictions.joblib') == result['prediction_sha256']
    source, val, allowed, vp, refs, origin = load_inputs(req['fold'])
    control, control_origin = load_control(req['fold'])
    assert origin == req['origin'] and control_origin == req['control']
    saved = joblib.load(root / 'predictions.joblib'); pred = {}; errors = dict(encoder=0., theta=0., refit=0., prediction=0., control=0., row=0., trial=0.)
    failures = []
    with threadpool_limits(limits=1):
        for spec in result['models']:
            kernel = spec['kernel']; assert sa.digest(Path(spec['path'])) == spec['sha256']
            old = joblib.load(spec['path'])
            fresh = RandomFunctionBranch.from_encoder(refs[kernel], source, allowed)
            assert fresh.source_keys == old.source_keys
            for attr in ('keep', 'latent_keep'): np.testing.assert_array_equal(getattr(fresh, attr), getattr(old, attr))
            for attr in ('raw_scaler', 'scaler'):
                for name in ('mean_', 'scale_'): np.testing.assert_array_equal(getattr(getattr(fresh, attr), name), getattr(getattr(old, attr), name))
            np.testing.assert_array_equal(fresh.pca.components_, old.pca.components_)
            errors['encoder'] = max(errors['encoder'], float(np.max(abs(fresh.gp.X_train_ - old.gp.X_train_))))
            errors['theta'] = max(errors['theta'], float(np.max(abs(fresh.gp.theta_ - old.gp.theta_))))
            assert fresh.gp.optimization_ == spec['optimization'] == old.gp.optimization_
            if not fresh.gp.optimization_['success']: failures.append(dict(kernel=kernel, optimization=spec['optimization']))
            slope_spec = next(m for m in control_origin['models'] if m['key'] == 'mixed__' + kernel)
            slope = joblib.load(slope_spec['path'])
            for c in val:
                masked = sa.inference_view(c); parts = {}
                for family, private in (('function_private', True), ('function_shared', False)):
                    p = independent_function_predict(fresh, masked, private)
                    parts[family] = p
                    errors['refit'] = max(errors['refit'], float(np.max(abs(ConditionalFunction(fresh, private).predict(masked) - saved[c.id, family + '__' + kernel]))))
                parts['function_uniform'] = (parts['function_private'] + parts['function_shared']) / 2
                parts['slope_uniform'] = (independent_predict(slope, masked, 'shared', point_mean=True) + independent_predict(slope, masked, 'private', point_mean=True)) / 2
                errors['control'] = max(errors['control'], float(np.max(abs(parts['slope_uniform'] - control[c.id, 'uniform__' + kernel]))))
                for family, p in parts.items():
                    key = family + '__' + kernel; pred[c.id, key] = p
                    errors['prediction'] = max(errors['prediction'], float(np.max(abs(p - saved[c.id, key]))))
            print(req['fold'], kernel, 'fresh refit and independent posterior replay', flush=True)
    assert set(pred) == set(saved) and len(result['models']) == 2
    byid = {c.id:c for c in val}; seen = set(); values = {}
    for row in result['rows']:
        cid, key, g, gain = row['cell_id'], row['key'], row['group'], row['gain']
        identity = (cid, key, g, gain); assert identity not in seen; seen.add(identity)
        c = byid[cid]; assert row['domain'] == c.domain and row['dataset'] == c.dataset
        p = (1 - gain) * vp[cid, g] + gain * pred[cid, key]
        loss = float(np.mean(abs(np.asarray(c.y[sa.K:], float) - p)))
        errors['row'] = max(errors['row'], abs(loss - row['mae']))
        values[identity] = loss
    expected = {(c.id, f + '__' + k, g, a) for c in val for f in req['protocol']['families'] for k in req['protocol']['kernels'] for g in PARENTS for a in req['protocol']['gains']}
    assert seen == expected
    objective = {}
    for t in result['trials']:
        domains = sorted({c.domain for c in val})
        means = [np.mean([values[c.id, t['key'], 'B123', t['gain']] for c in val if c.domain == d]) for d in domains]
        value = float(np.mean(means)); objective[t['key'], t['gain']] = value
        errors['trial'] = max(errors['trial'], abs(value - t['validation_mae']))
    assert len(objective) == 40
    for f, chosen in result['selected'].items():
        best = min([(k, a) for k, a in objective if k.startswith(f + '__')], key=lambda pair:(objective[pair], pair[1], pair[0]))
        assert best == (chosen['key'], chosen['gain'])
    assert max(errors.values()) < 1e-8, errors
    sa.write_json(root / 'verification.json', dict(status='REPLAY_PASS_OPTIMIZER_FAILURE_PRESENT' if failures else 'PASS',
        optimized_refits=2, rows=len(seen), trials=40, errors=errors, optimizer_failures=failures,
        result_sha256=sa.digest(root / 'result.json'), request_sha256=sa.digest(root / 'request.json'),
        code_sha256=sa.digest(Path(__file__)), helper_sha256=sa.digest(HERE / 'audit_conditional_mixed.py'),
        limits='Fresh source GP refit with audited encoder reuse, independent target algebra and validation arithmetic; not fresh encoder fit or outer performance evidence.'))
    print('PASS replay, optimizer warnings', failures, errors, flush=True)


if __name__ == '__main__':
    main()
