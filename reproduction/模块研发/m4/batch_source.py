"""Run source nesting, audit and validation selection before outer M4 scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
from source_oof import sa,HERE


def run(fold,out):
    folder=out/'folds'/fold.replace(':','__');folder.mkdir(parents=True,exist_ok=False)
    for script,args in [('source_oof.py',['--fold',fold,'--out',str(folder/'source')]),
                        ('audit_source_oof.py',['--root',str(folder/'source')]),
                        ('correction_pilot.py',['--source',str(folder/'source'),'--out',str(folder/'correction')])]:
        with (folder/(script+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            return dict(fold=fold,status='FAILED',script=script,exit_code=result.returncode)
    return dict(fold=fold,status='VALIDATION_SELECTED',source=str(folder/'source'),correction=str(folder/'correction'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=HERE.parent/'results';obj=json.loads((root/'m3_waveform_candidate_v1/candidate.json').read_text())
    folds=sorted({r['fold'] for r in obj['manifest']})
    pilot=root/'m4_source_oof_pilot_v1';corr=root/'m4_correction_pilot_v1'
    assert json.loads((pilot/'verification.json').read_text())['status']=='PASS'
    assert json.loads((corr/'result.json').read_text())['status']=='VALIDATION_PILOT_COMPLETE_NOT_ADOPTED'
    results=[dict(fold='MATR:b1',status='VALIDATION_SELECTED',source=str(pilot),correction=str(corr))]
    sa.write_json(out/'request.json',dict(status='RUNNING',folds=folds,workers=args.workers,
        purpose='Source nesting + independent replay + validation selection only. No outer query scoring.',
        code_hashes={p.name:sa.digest(p) for p in HERE.glob('*.py')},
        protocol_sha256=sa.digest(HERE/'protocol.json'),correction_protocol_sha256=sa.digest(HERE/'correction_protocol.json')))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,f,out) for f in folds if f!='MATR:b1']):
            result=future.result();results.append(result);sa.write_json(out/'progress.json',dict(completed=results,total=21))
            print(json.dumps(result),flush=True)
    sa.write_json(out/'result.json',dict(status='VALIDATION_BATCH_COMPLETE' if all(r['status']=='VALIDATION_SELECTED' for r in results) else 'BATCH_HAS_FAILURES',folds=results))


if __name__=='__main__':main()
