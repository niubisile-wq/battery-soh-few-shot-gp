"""All21 source/validation folds with fresh GP replay; never outer-score."""
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from source_oof import sa, HERE


def checked(root, fold, protocol):
    req = json.loads((root / 'request.json').read_text())
    result = json.loads((root / 'result.json').read_text())
    audit = json.loads((root / 'verification.json').read_text())
    assert req['fold'] == fold and req['protocol'] == protocol
    assert audit['status'] in ('PASS', 'REPLAY_PASS_OPTIMIZER_FAILURE_PRESENT')
    assert audit['result_sha256'] == sa.digest(root / 'result.json') and audit['request_sha256'] == sa.digest(root / 'request.json')
    assert audit['code_sha256'] == sa.digest(HERE / 'audit_random_function_fit.py')
    assert all(sa.digest(HERE / n) == h for n, h in audit['helper_hashes'].items())
    assert audit['optimized_refits'] == 5 and max(audit['errors'].values()) < 1e-8
    assert all(sa.digest(HERE / n) == h for n, h in req['code_hashes'].items())
    assert result['prediction_sha256'] == sa.digest(root / 'validation_predictions.joblib')
    assert len(result['models']) == 5 and [m['multiplier'] for m in result['models']] == protocol['initial_length_multipliers']
    warnings = []
    for model in result['models']:
        assert sa.digest(Path(model['path'])) == model['sha256']
        if not model['optimization']['success']:
            warnings.append(dict(multiplier=model['multiplier'], optimization=model['optimization']))
    assert warnings == audit['optimizer_failures']
    return dict(fold=fold, root=str(root), status='SOURCE_REPLAY_AUDITED', optimizer_failures=warnings,
                result_sha256=sa.digest(root / 'result.json'), audit_sha256=sa.digest(root / 'verification.json'))


def run(fold, out, protocol):
    folder = out / 'folds' / fold.replace(':', '__'); branch = folder / 'branch'
    try:
        folder.mkdir(parents=True, exist_ok=False)
        for script, args in [('random_function_fit_pilot.py', ['--fold', fold, '--out', str(branch)]),
                             ('audit_random_function_fit.py', ['--root', str(branch)])]:
            with (folder / (script + '.log')).open('w') as log:
                p = subprocess.run([sys.executable, str(HERE / script), *args], stdout=log, stderr=subprocess.STDOUT)
            if p.returncode:
                return dict(fold=fold, root=str(branch), status='TASK_FAILED', script=script, exit_code=p.returncode)
        return checked(branch, fold, protocol)
    except Exception as exc:
        return dict(fold=fold, root=str(branch), status='TASK_FAILED', error=repr(exc))


def main():
    results = HERE.parent / 'results'; pilot = results / 'm4_random_function_fit_pilot_v1'
    protocol = json.loads((HERE / 'random_function_fit_stability_protocol.json').read_text())
    done = [checked(pilot, 'MATR:b1', protocol)]
    source = results / 'm4_difference_batch_v1/result.json'
    previous = json.loads(source.read_text()); assert previous['status'] == 'VALIDATION_BATCH_COMPLETE'
    folds = sorted(r['fold'] for r in previous['folds']); assert len(folds) == len(set(folds)) == 21
    out = results / 'm4_random_function_fit_batch_v1'; out.mkdir(exist_ok=False)
    req = json.loads((pilot / 'request.json').read_text())
    sa.write_json(out / 'request.json', dict(folds=folds, workers=3, protocol=protocol, reused=done,
        source_batch_sha256=sa.digest(source), code_hashes={**req['code_hashes'], **{n:sa.digest(HERE / n) for n in
            ('batch_random_function_fit.py', 'audit_random_function_fit.py', 'audit_random_function.py', 'audit_conditional_mixed.py')}},
        limits='No auto outer query scoring. All21 validation choices and audits must complete before freeze. Failures retained, no automatic retry.'))
    sa.write_json(out / 'progress.json', dict(completed=done, total=21))
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(run, fold, out, protocol) for fold in folds if fold != 'MATR:b1']
        for future in as_completed(jobs):
            r = future.result(); done.append(r)
            sa.write_json(out / 'progress.json', dict(completed=done, total=21)); print(r, flush=True)
    assert len(done) == 21
    sa.write_json(out / 'result.json', dict(status='ALL_SOURCE_REPLAY_AUDITED' if all(r['status'] == 'SOURCE_REPLAY_AUDITED' for r in done) else 'BATCH_HAS_TASK_FAILURES',
        folds=done, optimizer_failure_count=sum(len(r.get('optimizer_failures', [])) for r in done)))


if __name__ == '__main__':
    main()

