"""Original-budget difference/direct matched branch, validation only."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from score import PARENTS
from difference_branch import DifferenceBranch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    results=HERE.parent/'results';protocol=json.loads((HERE/'difference_protocol.json').read_text())
    provpath=results/'m2_strict_ablation_v1/folds'/args.fold.replace(':','__')/'provenance.json'
    prov=json.loads(provpath.read_text());allowed={}
    for cid,j in prov['source_keys']:allowed.setdefault(cid,[]).append(j)
    train,val,test=sa.split(sa.load_cells(args.fold.split(':')[0]),args.fold)
    assert set(allowed)=={c.id for c in train} and set(allowed).isdisjoint(c.id for c in val+test)
    source=[sa.source_view(c,allowed[c.id]) for c in train]
    batch=json.loads((results/'m4_source_batch_v1/result.json').read_text())
    original=next(r for r in batch['folds'] if r['fold']==args.fold)
    parentpath=Path(original['correction'])/'validation_parents.joblib';vp=joblib.load(parentpath)
    assert set(vp)=={(c.id,g) for c in val for g in PARENTS}
    sa.write_json(out/'request.json',dict(fold=args.fold,protocol=protocol,source_keys=prov['source_keys'],
        provenance_sha256=sa.digest(provpath),validation_parent_path=str(parentpath),validation_parent_sha256=sa.digest(parentpath),
        code_hashes={n:sa.digest(HERE/n) for n in ('difference_pilot.py','difference_branch.py','difference_gp.py','multiscale.py','difference_protocol.json')}))
    predictions={};rows=[];trials=[];models=[];boundary=0.
    with threadpool_limits(limits=1):
        for target in protocol['targets']:
            for kernel in protocol['kernels']:
                key=target+'__'+kernel;model=DifferenceBranch(target,kernel).fit(source,allowed)
                assert set(model.source_keys)=={tuple(k) for k in prov['source_keys']}
                path=out/(key+'.joblib');joblib.dump(model,path,compress=3)
                models.append(dict(key=key,path=str(path),sha256=sa.digest(path),
                    optimization=getattr(model.gp,'optimization_',None),kernel=str(model.gp.kernel_)))
                for c in val:
                    p=model.predict(sa.inference_view(c));predictions[c.id,key]=p
                    yy=c.y.copy();yy[sa.K:]=999
                    boundary=max(boundary,float(np.max(abs(p-model.predict(replace(c,y=yy))))))
                    n=min(len(c.y),sa.K+3)
                    boundary=max(boundary,float(np.max(abs(p[:n-sa.K]-model.predict(sa.prefix(c,n))))))
                    for g in PARENTS:
                        for gain in protocol['gains']:
                            q=vp[c.id,g]+gain*(p-vp[c.id,g])
                            rows.append(dict(key=key,target=target,kernel=kernel,group=g,gain=gain,cell_id=c.id,
                                dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(q-c.y[sa.K:])))))
                for gain in protocol['gains']:
                    rr=[r for r in rows if r['key']==key and r['gain']==gain and r['group']=='B123']
                    trials.append(dict(target=target,key=key,gain=gain,validation_mae=sa.macro(rr)))
                print(args.fold,key,'source and validation complete',models[-1]['optimization'],flush=True)
    assert boundary<1e-8 and len(trials)==20
    selected={t:min([r for r in trials if r['target']==t],key=lambda r:(r['validation_mae'],r['gain'],r['key'])) for t in protocol['targets']}
    joblib.dump(predictions,out/'validation_branch_predictions.joblib',compress=3)
    sa.write_json(out/'result.json',dict(status='VALIDATION_COMPLETE_AUDIT_PENDING',selected=selected,rows=rows,trials=trials,
        models=models,boundary=boundary,prediction_sha256=sa.digest(out/'validation_branch_predictions.joblib')))
    print(selected,flush=True)


if __name__=='__main__':main()
