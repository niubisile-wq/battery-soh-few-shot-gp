"""Five source-GP initializations with frozen private0.5 configuration."""
import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from mixed_effects_pilot import load_inputs
from random_function_branch import RandomFunctionBranch, ConditionalFunction
from freeze_random_function_private_controls import build_manifest


def initialized_reference(reference, multiplier):
    if not np.isfinite(multiplier) or multiplier <= 0: raise ValueError('Invalid length multiplier')
    result = deepcopy(reference); kernel = result.gp.kernel
    names = [h.name for h in kernel.hyperparameters if not h.fixed]
    assert names == ['k1__k1__constant_value', 'k1__k2__length_scale', 'k2__noise_level']
    before = kernel.theta.copy(); theta = before.copy()
    theta[1] = np.clip(theta[1] + np.log(multiplier), *kernel.bounds[1])
    if multiplier != 1.: result.gp.kernel = kernel.clone_with_theta(theta)
    return result, dict(original_theta=before.tolist(), initial_theta=result.gp.kernel.theta.tolist(),
        clipped=bool(theta[1] != before[1] + np.log(multiplier)))


def load_case(fold):
    results = HERE.parent / 'results'; frozen_path = results / 'm4_random_function_private_controls_v1/frozen_selections.json'
    frozen = json.loads(frozen_path.read_text()); assert frozen == build_manifest(results)
    selection = next(s for s in frozen['selections'] if s['fold'] == fold)
    choice = selection['choices']['function_private']
    source, val, allowed, _, refs, origin = load_inputs(fold)
    key = choice['selected']['key'].split('__')[1]
    assert sa.digest(Path(choice['model']['path'])) == choice['model']['sha256']
    return source, val, allowed, refs[key], choice, dict(origin=origin, frozen_sha256=sa.digest(frozen_path))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    source, val, allowed, ref, choice, origin = load_case(args.fold)
    protocol = json.loads((HERE / 'random_function_fit_stability_protocol.json').read_text())
    sa.write_json(out / 'request.json', dict(fold=args.fold, choice=choice, origin=origin, protocol=protocol,
        code_hashes={n:sa.digest(HERE / n) for n in ('random_function_fit_pilot.py', 'random_function_fit_stability_protocol.json',
            'random_function_branch.py', 'random_function_gp.py', 'mixed_effects_pilot.py', 'mixed_effects_gp.py',
            'difference_gp.py', 'difference_branch.py', 'conditional_mixed.py', 'multiscale.py', 'freeze_random_function_private_controls.py')}))
    original = joblib.load(choice['model']['path']); models = []; predictions = {}; boundary = 0.; baseline = 0.
    with threadpool_limits(limits=1):
        for multiplier in protocol['initial_length_multipliers']:
            reference, initialization = initialized_reference(ref, multiplier)
            model = RandomFunctionBranch.from_encoder(reference, source, allowed)
            assert model.source_keys == original.source_keys
            if multiplier == 1.:
                baseline = max(baseline, float(np.max(abs(model.gp.theta_ - original.gp.theta_))))
                assert model.gp.optimization_ == original.gp.optimization_
            adapter = ConditionalFunction(model, True)
            for c in val:
                p = adapter.predict(sa.inference_view(c)); predictions[multiplier, c.id] = p
                yy = c.y.copy(); yy[sa.K:] = 999; n = min(sa.K + 3, len(yy))
                boundary = max(boundary, float(np.max(abs(p - adapter.predict(replace(c, y=yy))))),
                    float(np.max(abs(p[:n-sa.K] - adapter.predict(sa.prefix(c, n))))))
                if multiplier == 1.: baseline = max(baseline, float(np.max(abs(p - ConditionalFunction(original, True).predict(sa.inference_view(c))))))
            path = out / ('initial_' + str(multiplier) + '.joblib'); joblib.dump(model, path, compress=3)
            theta, bounds = model.gp.kernel_.theta, model.gp.kernel_.bounds
            hits = np.flatnonzero(np.any(np.isclose(theta[:, None], bounds, atol=1e-4, rtol=0), axis=1)).tolist()
            models.append(dict(multiplier=multiplier, path=str(path), sha256=sa.digest(path), initialization=initialization,
                optimization=model.gp.optimization_, diagnostics=model.gp.diagnostics_, kernel_bound_coordinates=hits,
                source_nll=float(model.gp.objective(model.gp.theta_, False))))
            print(args.fold, multiplier, model.gp.optimization_, flush=True)
    assert len(models) == 5 and max(boundary, baseline) < 1e-8
    joblib.dump(predictions, out / 'validation_predictions.joblib', compress=3)
    sa.write_json(out / 'result.json', dict(status='SOURCE_FITS_COMPLETE_AUDIT_PENDING', models=models, boundary=boundary,
        original_replay_error=baseline, prediction_sha256=sa.digest(out / 'validation_predictions.joblib'),
        limits='All five source fits retained. Validation predictions only for engineering replay, no loss-based selection or outer query scoring.'))


if __name__ == '__main__': main()
