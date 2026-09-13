"""Add missing evaluation_v3 import dependency; preserve failed first probe."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from source_oof import sa, HERE
from check_random_function_snapshot_runtime import PROBE


def main():
    root = HERE.parent / 'results/m4_random_function_candidate_v1'
    previous = json.loads((root / 'runtime_attempt.json').read_text())
    assert previous['exit_code'] != 0 and "cannot import name 'sa' from 'evaluate'" in previous['stderr']
    assert previous['probe_sha256'] == sa.digest(root / 'runtime_probe.joblib')
    assert not (root / 'runtime_attempt_v2.json').exists(), 'Preserve prior attempts'
    request = json.loads((root / 'request.json').read_text())
    assert all(sa.digest(root / n) == h for n, h in request['source_snapshot'].items())
    original = HERE.parent / 'evaluation_v3/evaluate.py'
    target = root / 'source_snapshot/evaluation_v3/evaluate.py'
    assert not target.exists(); target.parent.mkdir(parents=True, exist_ok=False); shutil.copy2(original, target)
    sa.write_json(root / 'runtime_snapshot_supplement.json', dict(
        reason='Original source_snapshot omitted evaluation_v3/evaluate.py imported by m3/screen.py; no model, prediction or original snapshot edits.',
        files=[dict(path=str(target.relative_to(root)), sha256=sa.digest(target), original_path=str(original), original_sha256=sa.digest(original))],
        failed_attempt_sha256=sa.digest(root / 'runtime_attempt.json'), code_sha256=sa.digest(Path(__file__))))
    with tempfile.TemporaryDirectory(prefix='m4-runtime-check-v2-') as temporary:
        run = subprocess.run([sys.executable, '-I', '-c', PROBE, str(root), str(HERE.parent)], cwd=temporary, capture_output=True, text=True)
    sa.write_json(root / 'runtime_attempt_v2.json', dict(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr,
        probe_sha256=sa.digest(root / 'runtime_probe.joblib'), supplement_sha256=sa.digest(root / 'runtime_snapshot_supplement.json'),
        code_sha256=sa.digest(Path(__file__)), probe_code_sha256=sa.digest(HERE / 'check_random_function_snapshot_runtime.py')))
    assert run.returncode == 0, run.stderr
    assert sa.digest(target) == sa.digest(original)
    assert all(sa.digest(root / n) == h for n, h in request['source_snapshot'].items())
    result = json.loads(run.stdout)
    result.update(candidate_sha256=sa.digest(root / 'candidate.json'), audit_sha256=sa.digest(root / 'verification.json'),
        attempt_sha256=sa.digest(root / 'runtime_attempt_v2.json'), supplement_sha256=sa.digest(root / 'runtime_snapshot_supplement.json'),
        limits='Three representative cells/24 adapters, fresh isolated interpreter using packaged project code plus installed numerical dependencies. Failed first probe retained; evaluation import dependency added without changing models.')
    sa.write_json(root / 'runtime_verification.json', result)
    print(result['status'], result['error'], flush=True)


if __name__ == '__main__': main()
