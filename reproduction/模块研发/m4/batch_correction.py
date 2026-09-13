"""Refit correction variants using existing audited nested source episodes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
from source_oof import sa,HERE


def run(entry,out,variant):
    folder=out/'folds'/entry['fold'].replace(':','__');folder.mkdir(parents=True,exist_ok=False)
    dest=folder/'correction'
    jobs=[('correction_pilot.py',['--source',entry['source'],'--out',str(dest),'--variant',variant]),
          ('audit_correction.py',['--source',entry['source'],'--correction',str(dest)])]
    for script,args in jobs:
        with (folder/(script+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:return dict(fold=entry['fold'],status='FAILED',script=script,exit_code=p.returncode)
    return dict(fold=entry['fold'],status='VALIDATION_SELECTED',source=entry['source'],correction=str(dest))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-batch',required=True);ap.add_argument('--out',required=True);ap.add_argument('--score-out',required=True);ap.add_argument('--variant',choices=['anchored','condition','consensus','robust','centered','within'],default='anchored');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    source=Path(args.source_batch).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    batch=json.loads((source/'result.json').read_text());assert batch['status']=='VALIDATION_BATCH_COMPLETE' and len(batch['folds'])==21
    sa.write_json(out/'request.json',dict(variant=args.variant,source_batch=str(source),source_batch_sha256=sa.digest(source/'result.json'),
        code_hashes={p.name:sa.digest(p) for p in HERE.glob('*.py')},protocol_sha256=sa.digest(HERE/(args.variant+'_protocol.json')),
        purpose='All21 validation selections before any new outer scoring; no source-GP refitting.'))
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run,e,out,args.variant) for e in batch['folds']]):
            result=f.result();results.append(result);sa.write_json(out/'progress.json',dict(completed=results,total=21));print(json.dumps(result),flush=True)
    success=all(r['status']=='VALIDATION_SELECTED' for r in results)
    sa.write_json(out/'result.json',dict(status='VALIDATION_BATCH_COMPLETE' if success else 'BATCH_HAS_FAILURES',folds=results))
    if success:subprocess.run([sys.executable,str(HERE/'score.py'),'--batch',str(out),'--out',str(Path(args.score_out).resolve())],check=True)


if __name__=='__main__':main()
