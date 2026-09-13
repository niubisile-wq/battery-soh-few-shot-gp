"""Complete original21 validation folds; all-four-source-model audit per fold."""
import json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from source_oof import sa,HERE


def run(fold,out):
    folder=out/'folds'/fold.replace(':','__');folder.mkdir(parents=True,exist_ok=False)
    branch=folder/'branch'
    for script,args in [('mixed_effects_pilot.py',['--fold',fold,'--out',str(branch)]),
                        ('audit_mixed_effects.py',['--root',str(branch)])]:
        with (folder/(script+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:return dict(fold=fold,status='FAILED',script=script,exit_code=p.returncode)
    return dict(fold=fold,status='VALIDATION_AUDITED',branch=str(branch))


def main():
    results=HERE.parent/'results';pilot=results/'m4_mixed_effects_pilot_v1'
    audit=json.loads((pilot/'verification.json').read_text());req=json.loads((pilot/'request.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(pilot/'result.json')
    assert audit['request_sha256']==sa.digest(pilot/'request.json') and audit['models_refitted']==4
    assert audit['code_sha256']==sa.digest(HERE/'audit_mixed_effects.py')
    assert req['protocol']==json.loads((HERE/'mixed_effects_protocol.json').read_text()) and req['fold']=='MATR:b1'
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    source=results/'m4_difference_batch_v1/result.json';previous=json.loads(source.read_text())
    assert previous['status']=='VALIDATION_BATCH_COMPLETE'
    folds=[r['fold'] for r in previous['folds']];assert len(folds)==len(set(folds))==21
    out=results/'m4_mixed_effects_batch_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'request.json',dict(folds=folds,workers=3,protocol=req['protocol'],reused_pilot=str(pilot),
        source_batch_sha256=sa.digest(source),code_hashes={**req['code_hashes'],
            'audit_mixed_effects.py':sa.digest(HERE/'audit_mixed_effects.py'),'batch_mixed_effects.py':sa.digest(Path(__file__))},
        scope='Original validation only. No query scoring auto-start.'))
    done=[dict(fold='MATR:b1',status='VALIDATION_AUDITED',branch=str(pilot))]
    sa.write_json(out/'progress.json',dict(completed=done,total=21))
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(run,f,out) for f in folds if f!='MATR:b1']
        for future in as_completed(jobs):
            r=future.result();done.append(r);sa.write_json(out/'progress.json',dict(completed=done,total=21));print(r,flush=True)
    sa.write_json(out/'result.json',dict(status='VALIDATION_BATCH_COMPLETE' if all(r['status']=='VALIDATION_AUDITED' for r in done) else 'BATCH_HAS_FAILURES',folds=done))


if __name__=='__main__':main()
