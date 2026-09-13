"""All21 validation-only support-gate selections before new outer scoring."""
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from support_gate import SupportGate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registered',action='store_true');args=ap.parse_args()
    prefix='m4_registered_gate' if args.registered else 'm4_support_gate'
    results=HERE.parent/'results';out=results/(prefix+'_validation_v1');out.mkdir(exist_ok=False)
    protocol=json.loads((HERE/'support_gate_protocol.json').read_text())
    batchpath=results/('m4_registered_batch_v1' if args.registered else 'm4_multiscale_batch_v1')/'result.json'
    batch=json.loads(batchpath.read_text())
    assert batch['status']=='VALIDATION_BATCH_COMPLETE'
    sa.write_json(out/'protocol.json',dict(protocol=protocol,registered=args.registered,source_batch=str(batchpath),source_batch_sha256=sa.digest(batchpath),code_sha256=sa.digest(Path(__file__)),gate_code_sha256=sa.digest(HERE/'support_gate.py')))
    selections=[]
    with threadpool_limits(limits=1):
        for e in batch['folds']:
            root=Path(e['branch']);req=json.loads((root/'request.json').read_text());r=json.loads((root/'result.json').read_text())
            audit=json.loads((root/'verification.json').read_text());assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
            vp=joblib.load(req['validation_parent_path']);bp=joblib.load(root/'validation_branch_predictions.joblib')
            assert sa.digest(Path(req['validation_parent_path']))==req['validation_parent_sha256']
            assert sa.digest(root/'validation_branch_predictions.joblib')==r['prediction_sha256']
            cells=sa.load_cells(e['fold'].split(':')[0]);_,val,_=sa.split(cells,e['fold'])
            dest=out/'folds'/e['fold'].replace(':','__');dest.mkdir(parents=True)
            trials=[];gates={};specs={}
            for spec in r['models']:
                assert sa.digest(Path(spec['path']))==spec['sha256']
                gp=joblib.load(spec['path']);gate=SupportGate(gp)
                stem=spec['encoder']+'__'+spec['kernel'];path=dest/(stem+'.joblib');joblib.dump(gate,path,compress=3)
                specs[stem]=dict(path=str(path),sha256=sa.digest(path),model=spec,scale2=gate.scale2)
                for c in val:
                    for multiplier in protocol['multipliers']:gates[c.id,stem,multiplier]=gate.predict(c,multiplier)
                for mode in req['protocol']['modes']:
                    key=stem+'__'+mode
                    for multiplier in protocol['multipliers']:
                        for gain in protocol['gains']:
                            bydomain={d:[] for d in {c.domain for c in val}}
                            for c in val:
                                p=vp[c.id,'B123'];q=bp[c.id,key]
                                pred=p+gain*gates[c.id,stem,multiplier]*(q-p)
                                bydomain[c.domain].append(float(np.mean(abs(pred-c.y[sa.K:]))))
                            trials.append(dict(key=key,multiplier=multiplier,gain=gain,validation_mae=float(np.mean([np.mean(v) for v in bydomain.values()]))))
            assert len(trials)==120
            chosen=min(trials,key=lambda t:(t['validation_mae'],t['gain'],t['key'],t['multiplier']))
            sa.write_json(dest/'result.json',dict(fold=e['fold'],selected=chosen,trials=trials,gates=specs,source_branch=str(root)))
            selections.append(dict(fold=e['fold'],selected=chosen,gate=specs['__'.join(chosen['key'].split('__')[:2])],
                                   result=str(dest/'result.json'),result_sha256=sa.digest(dest/'result.json')))
            sa.write_json(out/'progress.json',dict(completed=len(selections),total=21))
            print(e['fold'],'120validation trials frozen',flush=True)
    assert len(selections)==21
    sa.write_json(out/'result.json',dict(status='ALL_VALIDATION_SELECTED_AUDIT_PENDING',selections=selections,
        limits=protocol['limits']))


if __name__=='__main__':main()
