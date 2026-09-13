"""Wait for a verified live batch, then run scoring once; never restart training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--batch-pid',type=int,required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    batch=Path(args.batch).resolve();out=Path(args.out).resolve()
    assert not out.exists(),'Refuse to overwrite or duplicate an existing scoring run'
    previous=-1
    while not (batch/'result.json').exists():
        proc=Path('/proc')/str(args.batch_pid)/'cmdline'
        try: command=proc.read_bytes()
        except FileNotFoundError:raise RuntimeError('Batch process missing without final result; inspect, do not restart automatically')
        assert b'batch_source.py' in command,'Batch PID no longer identifies expected process'
        if (batch/'progress.json').exists():
            p=json.loads((batch/'progress.json').read_text());n=len(p['completed'])
            if n!=previous:print('Verified batch live, completed folds',n,'of21',flush=True);previous=n
        time.sleep(10)
    result=json.loads((batch/'result.json').read_text());assert result['status']=='VALIDATION_BATCH_COMPLETE',result['status']
    assert len(result['folds'])==21
    print('All21 folds terminal and selected; starting pre-score audits and16-group evaluation',flush=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('score.py')),'--batch',str(batch),'--out',str(out)],check=True)


if __name__=='__main__':main()
