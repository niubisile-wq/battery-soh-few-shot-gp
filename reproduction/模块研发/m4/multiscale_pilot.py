"""Budgeted direct multiscale branch: source fitting and validation only."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from multiscale import MultiscaleGP
from conditional_geometry import ConditionalGeometry
from score import PARENTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',required=True);ap.add_argument('--out',required=True);variants=ap.add_mutually_exclusive_group();variants.add_argument('--structured',action='store_true');variants.add_argument('--registered',action='store_true')
    args=ap.parse_args();fold=args.fold;out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    results=HERE.parent/'results';protocol=json.loads((HERE/('registered_protocol.json' if args.registered else 'structured_protocol.json' if args.structured else 'multiscale_protocol.json')).read_text())
    estimator=MultiscaleGP
    if args.structured:
        from structured_multiscale import StructuredMultiscaleGP
        estimator=StructuredMultiscaleGP
    if args.registered:
        from registered_charge import RegisteredChargeGP
        estimator=RegisteredChargeGP
    provpath=results/'m2_strict_ablation_v1/folds'/fold.replace(':','__')/'provenance.json'
    prov=json.loads(provpath.read_text());allowed={}
    for cid,j in prov['source_keys']:allowed.setdefault(cid,[]).append(j)
    cells=sa.load_cells(fold.split(':')[0]);train,validation,test=sa.split(cells,fold)
    assert set(allowed)=={c.id for c in train}
    assert set(allowed).isdisjoint({c.id for c in validation+test})
    assert sum(map(len,allowed.values()))<=1000
    visible=[sa.source_view(c,allowed[c.id]) for c in train]
    # Frozen validation parents already have independent source/refit audits.
    batch=json.loads((results/'m4_source_batch_v1/result.json').read_text())
    original=next(r for r in batch['folds'] if r['fold']==fold)
    parentpath=Path(original['correction'])/'validation_parents.joblib'
    vp=joblib.load(parentpath)
    assert set(vp)=={(c.id,g) for c in validation for g in PARENTS}
    sa.write_json(out/'request.json',dict(fold=fold,structured=args.structured,registered=args.registered,protocol=protocol,source_keys=prov['source_keys'],
        provenance_sha256=sa.digest(provpath),validation_parent_path=str(parentpath),validation_parent_sha256=sa.digest(parentpath),
        code_hashes={p.name:sa.digest(p) for p in [Path(__file__),HERE/'multiscale.py']+([HERE/'structured_multiscale.py'] if args.structured else [])+([HERE/'registered_charge.py'] if args.registered else [])},
        scope='Source fitting and original validation only; no held-domain query scoring.'))
    trials=[];rows=[];manifest=[];predictions={};boundary=0.
    with threadpool_limits(limits=1):
        for encoder in protocol['encoders']:
            for kernel in protocol['kernels']:
                gp=estimator(encoder,kernel).fit(visible,allowed)
                assert set(gp.source_keys)=={tuple(k) for k in prov['source_keys']}
                modelpath=out/'models'/(encoder+'__'+kernel+'.joblib');modelpath.parent.mkdir(exist_ok=True)
                joblib.dump(gp,modelpath,compress=3)
                manifest.append(dict(encoder=encoder,kernel=kernel,path=str(modelpath),sha256=sa.digest(modelpath),source_keys=gp.source_keys))
                for mode in protocol['modes']:
                    key='__'.join((encoder,kernel,mode));branch=ConditionalGeometry(gp,mode)
                    for c in validation:
                        pred=branch.predict(sa.inference_view(c));predictions[c.id,key]=pred
                        yy=c.y.copy();yy[sa.K:]=999
                        boundary=max(boundary,float(np.max(abs(pred-branch.predict(replace(c,y=yy))))))
                        n=min(len(c.x),sa.K+3)
                        boundary=max(boundary,float(np.max(abs(pred[:n-sa.K]-branch.predict(sa.prefix(c,n))))))
                        for g in PARENTS:
                            parent=vp[c.id,g]
                            for gain in protocol['gains']:
                                prediction=parent+gain*(pred-parent)
                                rows.append(dict(key=key,group=g,gain=gain,cell_id=c.id,dataset=c.dataset,domain=c.domain,
                                    mae=float(np.mean(abs(prediction-c.y[sa.K:])))))
                    for gain in protocol['gains']:
                        rr=[r for r in rows if r['key']==key and r['gain']==gain and r['group']=='B123']
                        trials.append(dict(key=key,gain=gain,validation_mae=sa.macro(rr)))
                print(fold,encoder,kernel,'source fit and both validation modes complete',flush=True)
    assert boundary<1e-8 and len(trials)==40 and len(manifest)==4
    selected=min(trials,key=lambda r:(r['validation_mae'],r['gain'],r['key']))
    joblib.dump(predictions,out/'validation_branch_predictions.joblib',compress=3)
    sa.write_json(out/'result.json',dict(status='VALIDATION_COMPLETE_AUDIT_PENDING',selected=selected,trials=trials,
        rows=rows,models=manifest,max_boundary_error=boundary,
        identity_validation_mae=min(t['validation_mae'] for t in trials if t['gain']==0),
        prediction_sha256=sa.digest(out/'validation_branch_predictions.joblib'),limits=protocol['limits']))
    print(json.dumps(dict(selected=selected,boundary=boundary)),flush=True)


if __name__=='__main__':main()
