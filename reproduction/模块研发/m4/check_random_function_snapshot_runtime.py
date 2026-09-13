"""Fresh isolated interpreter, only packaged project code, 3 representative cells."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import joblib
from source_oof import sa, HERE


PROBE = r'''
import json,sys
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
root=Path(sys.argv[1]);snapshot=root/'source_snapshot';project=Path(sys.argv[2])
sys.path.insert(0,str(snapshot/'m4'))
import random_function_module
assert Path(random_function_module.__file__).is_relative_to(snapshot)
samples=joblib.load(root/'runtime_probe.joblib');error=0.;count=0
with threadpool_limits(limits=1):
    for sample in samples:
        c=sample['cell'];assert np.isnan(c.y[10:]).all()
        with np.load(root/sample['predictions']) as z:
            for entry in sample['adapters']:
                model=joblib.load(root/entry['artifact']);p=model.predict(c)
                error=max(error,float(np.max(abs(p-z[entry['group']]))));count+=1
imports={}
for name,module in tuple(sys.modules.items()):
    path=getattr(module,'__file__',None)
    if path and Path(path).is_relative_to(project):
        assert Path(path).is_relative_to(snapshot),(name,path)
        imports[name]=str(Path(path).relative_to(root))
assert count==24 and error<1e-8
print(json.dumps(dict(status='ISOLATED_SNAPSHOT_RUNTIME_PASS',cells=3,adapters=24,error=error,project_imports=imports)))
'''


def main():
    root = HERE.parent / 'results/m4_random_function_candidate_v1'
    candidate = json.loads((root / 'candidate.json').read_text()); audit = json.loads((root / 'verification.json').read_text())
    assert audit['status'] == 'EXPORTED_CANDIDATE_AUDIT_PASS' and audit['candidate_sha256'] == sa.digest(root / 'candidate.json')
    assert not (root / 'runtime_probe.joblib').exists(), 'Do not overwrite an existing probe attempt'
    samples = []
    for ds in ('XJTU', 'MATR', 'Tongji'):
        c = min(sa.load_cells(ds), key=lambda c:c.id); fold = ds + ':' + c.domain
        entries = [r for r in candidate['manifest'] if r['fold'] == fold]; assert len(entries) == 8
        prediction = next(r for r in candidate['predictions'] if r['cell_id'] == c.id)
        samples.append(dict(cell=sa.inference_view(c), adapters=entries, predictions=prediction['path']))
    joblib.dump(samples, root / 'runtime_probe.joblib', compress=3)
    with tempfile.TemporaryDirectory(prefix='m4-runtime-check-') as temporary:
        run = subprocess.run([sys.executable, '-I', '-c', PROBE, str(root), str(HERE.parent)],
            cwd=temporary, capture_output=True, text=True)
    sa.write_json(root / 'runtime_attempt.json', dict(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr,
        probe_sha256=sa.digest(root / 'runtime_probe.joblib'), code_sha256=sa.digest(Path(__file__))))
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    result.update(candidate_sha256=sa.digest(root / 'candidate.json'), audit_sha256=sa.digest(root / 'verification.json'),
        attempt_sha256=sa.digest(root / 'runtime_attempt.json'),
        limits='Three representative cells across datasets,24 exported adapters. Only project code isolation; installed numerical dependencies still required. Not fresh training or new generalization evidence.')
    sa.write_json(root / 'runtime_verification.json', result)
    print(result['status'], result['error'], flush=True)


if __name__ == '__main__': main()
