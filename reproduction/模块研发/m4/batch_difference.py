"""Twenty remaining original validation folds; no automatic held-domain scoring."""
import json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from source_oof import sa,HERE


def run(fold,out):
    folder=out/'folds'/fold.replace(':','__');folder.mkdir(parents=True,exist_ok=False)
    branch=folder/'branch'
    for script,args in [('difference_pilot.py',['--fold',fold,'--out',str(branch)]),
                        ('audit_difference.py',['--root',str(branch)])]:
        with (folder/(script+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:return dict(fold=fold,status='FAILED',script=script,exit_code=p.returncode)
    return dict(fold=fold,status='VALIDATION_AUDITED',branch=str(branch))


def main():
    results=HERE.parent/'results';pilot=results/'m4_difference_pilot_v1'
    audit=json.loads((pilot/'verification.json').read_text());req=json.loads((pilot/'request.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(pilot/'result.json')
    assert audit['request_sha256']==sa.digest(pilot/'request.json')
    assert audit['code_sha256']==sa.digest(HERE/'audit_difference.py')
    assert req['protocol']==json.loads((HERE/'difference_protocol.json').read_text())
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    assert req['fold']=='MATR:b1'
    folds=[r['fold'] for r in json.loads((results/'m4_source_batch_v1/result.json').read_text())['folds']]
    assert len(set(folds))==len(folds)==21
    out=results/'m4_difference_batch_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'request.json',dict(folds=folds,protocol=req['protocol'],reused_pilot=str(pilot),workers=3,
        code_hashes={**req['code_hashes'],'audit_difference.py':sa.digest(HERE/'audit_difference.py'),'batch_difference.py':sa.digest(Path(__file__))}))
    done=[dict(fold='MATR:b1',status='VALIDATION_AUDITED',branch=str(pilot))]
    sa.write_json(out/'progress.json',dict(completed=done,total=21))
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(run,f,out) for f in folds if f!='MATR:b1']
        for future in as_completed(jobs):
            r=future.result();done.append(r);sa.write_json(out/'progress.json',dict(completed=done,total=21));print(r,flush=True)
    sa.write_json(out/'result.json',dict(status='VALIDATION_BATCH_COMPLETE' if all(r['status']=='VALIDATION_AUDITED' for r in done) else 'BATCH_HAS_FAILURES',folds=done))


if __name__=='__main__':main()
