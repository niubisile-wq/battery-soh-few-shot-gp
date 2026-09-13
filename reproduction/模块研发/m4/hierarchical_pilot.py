"""Two-level versus cell-only source fits, original validation only."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from mixed_effects_pilot import load_inputs
from hierarchical_branch import fit_branch
from score import PARENTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((HERE/'hierarchical_protocol.json').read_text())
    source,val,allowed,vp,refs,origin=load_inputs(args.fold)
    previous=HERE.parent/'results/m4_evidence_validation_v1/folds'/args.fold.replace(':','__')
    old=json.loads((previous/'result.json').read_text())
    assert sa.digest(previous/'predictions.joblib')==old['prediction_sha256']
    cached=joblib.load(previous/'predictions.joblib')['predictions']
    sa.write_json(out/'request.json',dict(fold=args.fold,protocol=protocol,origin=origin,control_root=str(previous),
        control_result_sha256=sa.digest(previous/'result.json'),
        code_hashes={n:sa.digest(HERE/n) for n in ('hierarchical_pilot.py','hierarchical_branch.py','hierarchical_slopes.py','hierarchical_protocol.json',
            'mixed_effects_pilot.py','mixed_effects_gp.py','difference_gp.py','difference_branch.py','conditional_mixed.py','evidence_mixed.py')}))
    rows=[];trials=[];pred={};models=[];boundary=0.;control_error=0.
    with threadpool_limits(limits=1):
        for family in protocol['families']:
            for kernel in protocol['kernels']:
                key=family+'__'+kernel;model=fit_branch(refs[kernel],source,allowed,family=='hierarchical')
                path=out/(key+'.joblib');joblib.dump(model,path,compress=3)
                gp=model.parent.gp
                models.append(dict(key=key,path=str(path),sha256=sa.digest(path),optimization=gp.optimization_,diagnostics=gp.diagnostics_))
                for c in val:
                    q=model.predict(sa.inference_view(c));pred[c.id,key]=q
                    if family=='cell_only':control_error=max(control_error,float(np.max(abs(q-cached[c.id,'uniform__'+kernel]))))
                    yy=c.y.copy();yy[sa.K:]=999;n=min(len(c.y),sa.K+3)
                    boundary=max(boundary,float(np.max(abs(q-model.predict(replace(c,y=yy))))),
                        float(np.max(abs(q[:n-sa.K]-model.predict(sa.prefix(c,n))))))
                    for g in PARENTS:
                        for gain in protocol['gains']:
                            p=vp[c.id,g]+gain*(q-vp[c.id,g])
                            rows.append(dict(key=key,target=family,gain=gain,group=g,cell_id=c.id,dataset=c.dataset,domain=c.domain,
                                mae=float(np.mean(abs(p-c.y[sa.K:])))))
                for gain in protocol['gains']:
                    rr=[r for r in rows if r['key']==key and r['gain']==gain and r['group']=='B123']
                    trials.append(dict(target=family,key=key,gain=gain,validation_mae=sa.macro(rr)))
                print(args.fold,key,gp.optimization_,gp.diagnostics_,flush=True)
    assert len(trials)==20 and max(boundary,control_error)<1e-8
    selected={f:min([t for t in trials if t['target']==f],key=lambda t:(t['validation_mae'],t['gain'],t['key'])) for f in protocol['families']}
    assert selected['cell_only']['key'].replace('cell_only__','uniform__')==old['selected']['uniform']['key']
    assert selected['cell_only']['gain']==old['selected']['uniform']['gain']
    joblib.dump(pred,out/'predictions.joblib',compress=3)
    sa.write_json(out/'result.json',dict(status='VALIDATION_COMPLETE_AUDIT_PENDING',selected=selected,trials=trials,rows=rows,models=models,
        boundary=boundary,control_error=control_error,prediction_sha256=sa.digest(out/'predictions.joblib')))
    print(selected,flush=True)


if __name__=='__main__':main()
