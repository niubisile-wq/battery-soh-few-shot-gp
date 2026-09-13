"""Fresh five-init source refits plus independent validation target algebra."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from random_function_fit_pilot import load_case, initialized_reference
from random_function_branch import RandomFunctionBranch, ConditionalFunction
from audit_random_function import independent_function_predict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); root = Path(ap.parse_args().root).resolve()
    req = json.loads((root / 'request.json').read_text()); result = json.loads((root / 'result.json').read_text())
    assert all(sa.digest(HERE / n) == h for n, h in req['code_hashes'].items())
    assert sa.digest(root / 'validation_predictions.joblib') == result['prediction_sha256']
    source, val, allowed, ref, choice, origin = load_case(req['fold'])
    assert choice == req['choice'] and origin == req['origin']
    original = joblib.load(choice['model']['path']); cached = joblib.load(root / 'validation_predictions.joblib')
    errors = dict(encoder=0., theta=0., prediction=0., refit=0., baseline=0., nll=0.); failures = []; seen = set()
    assert [m['multiplier'] for m in result['models']] == req['protocol']['initial_length_multipliers']
    with threadpool_limits(limits=1):
        for spec in result['models']:
            scale = spec['multiplier']; assert sa.digest(Path(spec['path'])) == spec['sha256']
            old = joblib.load(spec['path']); reference, init = initialized_reference(ref, scale)
            assert init == spec['initialization']
            expected_length = np.clip(ref.gp.kernel.theta[1] + np.log(scale), *ref.gp.kernel.bounds[1])
            assert abs(reference.gp.kernel.theta[1] - expected_length) < 1e-14
            np.testing.assert_array_equal(reference.gp.kernel.theta[[0, 2]], ref.gp.kernel.theta[[0, 2]])
            fresh = RandomFunctionBranch.from_encoder(reference, source, allowed)
            assert fresh.source_keys == old.source_keys == original.source_keys
            for attr in ('keep', 'latent_keep'): np.testing.assert_array_equal(getattr(fresh, attr), getattr(old, attr))
            for attr in ('raw_scaler', 'scaler'):
                for name in ('mean_', 'scale_'): np.testing.assert_array_equal(getattr(getattr(fresh, attr), name), getattr(getattr(old, attr), name))
            np.testing.assert_array_equal(fresh.pca.components_, old.pca.components_)
            errors['encoder'] = max(errors['encoder'], float(np.max(abs(fresh.gp.X_train_ - old.gp.X_train_))))
            errors['theta'] = max(errors['theta'], float(np.max(abs(fresh.gp.theta_ - old.gp.theta_))))
            errors['nll'] = max(errors['nll'], abs(float(fresh.gp.objective(fresh.gp.theta_, False)) - spec['source_nll']))
            assert fresh.gp.optimization_ == old.gp.optimization_ == spec['optimization']
            if not spec['optimization']['success']: failures.append(dict(multiplier=scale, optimization=spec['optimization']))
            if scale == 1.: errors['baseline'] = max(errors['baseline'], float(np.max(abs(fresh.gp.theta_ - original.gp.theta_))))
            for c in val:
                visible = sa.inference_view(c)
                p = independent_function_predict(fresh, visible, True); seen.add((scale, c.id))
                errors['prediction'] = max(errors['prediction'], float(np.max(abs(p - cached[scale, c.id]))))
                errors['refit'] = max(errors['refit'], float(np.max(abs(ConditionalFunction(fresh, True).predict(visible) - cached[scale, c.id]))))
                if scale == 1.: errors['baseline'] = max(errors['baseline'], float(np.max(abs(p - ConditionalFunction(original, True).predict(visible)))))
            print(req['fold'], scale, 'fresh source refit independently checked', flush=True)
    assert seen == set(cached) and max(errors.values()) < 1e-8, errors
    sa.write_json(root / 'verification.json', dict(status='REPLAY_PASS_OPTIMIZER_FAILURE_PRESENT' if failures else 'PASS',
        optimized_refits=5, errors=errors, optimizer_failures=failures, result_sha256=sa.digest(root / 'result.json'),
        request_sha256=sa.digest(root / 'request.json'), code_sha256=sa.digest(Path(__file__)),
        helper_hashes={n:sa.digest(HERE / n) for n in ('audit_random_function.py', 'audit_conditional_mixed.py')},
        limits='Fresh source-GP refits on reused audited encoder; independent validation predictions, not outer performance or full pipeline retraining.'))
    print('PASS five-init replay', errors, 'optimizer failures', failures, flush=True)


if __name__ == '__main__': main()
