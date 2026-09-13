"""All21 multiscale validation folds, reusing only audited original pilot."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from source_oof import sa,HERE


def run(fold,out,structured=False,registered=False):
    folder=out/'folds'/fold.replace(':','__');folder.mkdir(parents=True,exist_ok=False)
    branch=folder/'branch'
    for script,args in [('multiscale_pilot.py',['--fold',fold,'--out',str(branch)]+(['--structured'] if structured else ['--registered'] if registered else [])),
                        ('audit_multiscale.py',['--root',str(branch)])]:
        with (folder/(script+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:return dict(fold=fold,status='FAILED',script=script,exit_code=p.returncode)
    return dict(fold=fold,status='VALIDATION_AUDITED',branch=str(branch))


def main():
    ap=argparse.ArgumentParser();variants=ap.add_mutually_exclusive_group();variants.add_argument('--structured',action='store_true');variants.add_argument('--registered',action='store_true');args=ap.parse_args()
    variant='registered' if args.registered else 'structured' if args.structured else 'multiscale'
    results=HERE.parent/'results';out=results/('m4_'+variant+'_batch_v1');out.mkdir(exist_ok=False)
    source=json.loads((results/'m4_source_batch_v1/result.json').read_text())
    folds=[r['fold'] for r in source['folds']];assert len(set(folds))==21
    pilot=results/('m4_'+variant+'_pilot_v1');audit=json.loads((pilot/'verification.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(pilot/'result.json')
    assert audit['code_sha256']==sa.digest(HERE/'audit_multiscale.py')
    request=json.loads((pilot/'request.json').read_text())
    assert request['protocol']==json.loads((HERE/(variant+'_protocol.json')).read_text())
    assert request.get('structured',False)==args.structured
    assert request.get('registered',False)==args.registered
    assert all(sa.digest(HERE/name)==digest for name,digest in request['code_hashes'].items())
    sa.write_json(out/'request.json',dict(folds=folds,reused_pilot=str(pilot),workers=3,
        code_hashes={p.name:sa.digest(p) for p in [HERE/'multiscale.py',HERE/'multiscale_pilot.py',HERE/'audit_multiscale.py',Path(__file__)]},
        protocol=request['protocol'],purpose='All validation choices before outer query scoring. No scoring auto-start in this runner.'))
    done=[dict(fold='MATR:b1',status='VALIDATION_AUDITED',branch=str(pilot))]
    sa.write_json(out/'progress.json',dict(completed=done,total=21))
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(run,fold,out,args.structured,args.registered) for fold in folds if fold!='MATR:b1']
        for future in as_completed(jobs):
            r=future.result();done.append(r);sa.write_json(out/'progress.json',dict(completed=done,total=21));print(json.dumps(r),flush=True)
    sa.write_json(out/'result.json',dict(status='VALIDATION_BATCH_COMPLETE' if all(r['status']=='VALIDATION_AUDITED' for r in done) else 'BATCH_HAS_FAILURES',folds=done))


if __name__=='__main__':main()
