"""One inner domain, fresh encoder and two source GPs; no outer query reads."""
import argparse,json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from difference_branch import DifferenceBranch
from mixed_effects_branch import MixedEffectsBranch
from support_cv_mixed import SupportCVMixed
from score import PARENTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',required=True);ap.add_argument('--held',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    results=HERE.parent/'results';feas=results/'m4_source_domain_feasibility_v1/summary.json'
    feasibility=json.loads(feas.read_text());assert feasibility['status']=='BUDGET_AND_CACHE_COVERAGE_VERIFIED'
    origin=next(r for r in feasibility['provenance'] if r['fold']==args.fold);source=Path(origin['root'])
    assert sa.digest(source/'complete_parent_source_oof.joblib')==origin['cache_sha256']
    assert sa.digest(source/'request.json')==origin['request_sha256']
    assert sa.digest(source/'audit_manifest.json')==origin['manifest_sha256']
    assert sa.digest(source/'verification.json')==origin['audit_sha256']
    budget=json.loads((source/'request.json').read_text())['source_keys'];allowed=defaultdict(list)
    for cid,j in budget:allowed[cid].append(j)
    train,_,_=sa.split(sa.load_cells(args.fold.split(':')[0]),args.fold)
    fit=[c for c in train if c.domain!=args.held];held={c.id:c for c in train if c.domain==args.held}
    assert held and fit and len(held)+len(fit)==len(train)
    fit_allowed={c.id:sorted(allowed[c.id]) for c in fit};visible=[sa.source_view(c,fit_allowed[c.id]) for c in fit]
    episodes=[e for e in joblib.load(source/'complete_parent_source_oof.joblib') if e['domain']==args.held]
    assert {e['cell_id'] for e in episodes}==set(held)
    protocol=json.loads((HERE/'source_domain_protocol.json').read_text())
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    code_names=('source_domain_pilot.py','source_domain_protocol.json','difference_branch.py','difference_gp.py',
        'mixed_effects_branch.py','mixed_effects_gp.py','support_cv_mixed.py','conditional_mixed.py','multiscale.py')
    sa.write_json(out/'request.json',dict(fold=args.fold,held=args.held,protocol=protocol,origin=origin,
        feasibility_sha256=sa.digest(feas),fit_keys=sorted((cid,j) for cid,ii in fit_allowed.items() for j in ii),
        code_hashes={n:sa.digest(HERE/n) for n in code_names}))
    models=[];rows=[];predictions={};boundary=0.
    with threadpool_limits(limits=1):
        for kernel in protocol['kernels']:
            reference=DifferenceBranch('difference',kernel,optimize=False).fit(visible,fit_allowed)
            model=MixedEffectsBranch.from_encoder(reference,visible,fit_allowed,optimize=True)
            assert set(model.source_keys)=={(cid,j) for cid,ii in fit_allowed.items() for j in ii}
            assert {cid for cid,j in model.source_keys}.isdisjoint(held)
            path=out/(kernel+'.joblib');joblib.dump(model,path,compress=3)
            models.append(dict(kernel=kernel,path=str(path),sha256=sa.digest(path),optimization=model.gp.optimization_,diagnostics=model.gp.diagnostics_))
            for e in episodes:
                c=held[e['cell_id']];indices=np.r_[np.arange(sa.K),e['query_indices']]
                assert set(e['query_indices'])=={j for j in allowed[c.id] if j>=sa.K}
                sparse=replace(c,x=c.x[indices],y=c.y[indices],cycle=c.cycle[indices]);masked=sa.inference_view(sparse)
                for family in protocol['families']:
                    adapter=SupportCVMixed(model,family);q=adapter.predict(masked)
                    yy=sparse.y.copy();yy[sa.K:]=999;n=min(sa.K+3,len(yy))
                    boundary=max(boundary,float(np.max(abs(q-adapter.predict(replace(sparse,y=yy))))),
                        float(np.max(abs(q[:n-sa.K]-adapter.predict(sa.prefix(sparse,n))))))
                    key=family+'__'+kernel;predictions[c.id,key]=q
                    for g in PARENTS:
                        for gain in protocol['gains']:
                            p=e['predictions'][g]+gain*(q-e['predictions'][g])
                            rows.append(dict(cell_id=c.id,domain=c.domain,key=key,group=g,gain=gain,
                                mae=float(np.mean(abs(p-e['y'])))))
            print(args.fold,args.held,kernel,model.gp.optimization_,model.gp.diagnostics_,flush=True)
    assert boundary<1e-8
    joblib.dump(predictions,out/'predictions.joblib',compress=3)
    sa.write_json(out/'result.json',dict(status='INNER_DOMAIN_COMPLETE_AUDIT_PENDING',models=models,rows=rows,boundary=boundary,
        prediction_sha256=sa.digest(out/'predictions.joblib'),optimized_fits=2,nonoptimized_reference_fits=2,
        limits='Single source-held domain,no outer candidate selection or query scoring. Independent encoder/source refit audit pending.'))


if __name__=='__main__':main()
