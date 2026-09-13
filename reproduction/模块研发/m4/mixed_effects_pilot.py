"""Audited encoder reuse, source mixed/zero covariance fitting, validation only."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from score import PARENTS
from multiscale import local_features
from mixed_effects_branch import MixedEffectsBranch


def load_inputs(fold):
    results=HERE.parent/'results'
    batch=json.loads((results/'m4_difference_batch_v1/result.json').read_text())
    assert batch['status']=='VALIDATION_BATCH_COMPLETE'
    entry=next(r for r in batch['folds'] if r['fold']==fold);root=Path(entry['branch'])
    req=json.loads((root/'request.json').read_text());res=json.loads((root/'result.json').read_text())
    audit=json.loads((root/'verification.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
    assert audit['request_sha256']==sa.digest(root/'request.json') and audit['refitted_models']==4
    assert audit['code_sha256']==sa.digest(HERE/'audit_difference.py')
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    provpath=results/'m2_strict_ablation_v1/folds'/fold.replace(':','__')/'provenance.json'
    assert sa.digest(provpath)==req['provenance_sha256']
    assert json.loads(provpath.read_text())['source_keys']==req['source_keys']
    train,val,test=sa.split(sa.load_cells(fold.split(':')[0]),fold);allowed={}
    for cid,j in req['source_keys']:allowed.setdefault(cid,[]).append(j)
    assert set(allowed)=={c.id for c in train} and set(allowed).isdisjoint(c.id for c in val+test)
    source=[sa.source_view(c,allowed[c.id]) for c in train];byid={c.id:c for c in source}
    refs={};specs={};latent_error=0.
    for kernel in ('rbf','matern'):
        spec=next(m for m in res['models'] if m['key']=='difference__'+kernel)
        assert sa.digest(Path(spec['path']))==spec['sha256'];m=joblib.load(spec['path'])
        assert set(m.source_keys)==set(map(tuple,req['source_keys']))
        features={cid:local_features(c) for cid,c in byid.items()}
        x=np.array([features[cid][j] for cid,j in m.source_keys])
        z=m.scaler.transform(m.pca.transform(m.raw_scaler.transform(x[:,m.keep]))[:,m.latent_keep])
        latent_error=max(latent_error,float(np.max(abs(z-m.gp.X_train_))))
        refs[kernel]=m;specs[kernel]=spec
    assert latent_error<1e-10
    parent=Path(req['validation_parent_path']);assert sa.digest(parent)==req['validation_parent_sha256']
    vp=joblib.load(parent);assert set(vp)=={(c.id,g) for c in val for g in PARENTS}
    return source,val,allowed,vp,refs,dict(reference_root=str(root),reference_request_sha256=sa.digest(root/'request.json'),
        reference_result_sha256=sa.digest(root/'result.json'),reference_audit_sha256=sa.digest(root/'verification.json'),
        source_keys=req['source_keys'],encoders=specs,latent_replay_error=latent_error,
        validation_parent_path=str(parent),validation_parent_sha256=sa.digest(parent))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((HERE/'mixed_effects_protocol.json').read_text())
    source,val,allowed,vp,refs,origin=load_inputs(args.fold)
    sa.write_json(out/'request.json',dict(fold=args.fold,protocol=protocol,origin=origin,
        code_hashes={n:sa.digest(HERE/n) for n in ('mixed_effects_pilot.py','mixed_effects_branch.py','mixed_effects_gp.py','mixed_effects_protocol.json','difference_branch.py','difference_gp.py','multiscale.py')}))
    predictions={};rows=[];trials=[];models=[];boundary=0.;zero_error=0.
    with threadpool_limits(limits=1):
        for family in protocol['families']:
            for kernel in protocol['kernels']:
                key=family+'__'+kernel
                m=MixedEffectsBranch.from_encoder(refs[kernel],source,allowed,learn_rho=family=='mixed')
                path=out/(key+'.joblib');joblib.dump(m,path,compress=3)
                models.append(dict(key=key,path=str(path),sha256=sa.digest(path),optimization=m.gp.optimization_,diagnostics=m.gp.diagnostics_))
                for c in val:
                    p=m.predict(sa.inference_view(c));predictions[c.id,key]=p
                    if family=='zero':zero_error=max(zero_error,float(np.max(abs(p-refs[kernel].predict(sa.inference_view(c))))))
                    yy=c.y.copy();yy[sa.K:]=999;n=min(len(c.y),sa.K+3)
                    boundary=max(boundary,float(np.max(abs(p-m.predict(replace(c,y=yy))))),
                        float(np.max(abs(p[:n-sa.K]-m.predict(sa.prefix(c,n))))))
                    for g in PARENTS:
                        for gain in protocol['gains']:
                            q=vp[c.id,g]+gain*(p-vp[c.id,g])
                            rows.append(dict(key=key,target=family,kernel=kernel,group=g,gain=gain,cell_id=c.id,
                                dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(q-c.y[sa.K:])))))
                for gain in protocol['gains']:
                    rr=[r for r in rows if r['key']==key and r['gain']==gain and r['group']=='B123']
                    trials.append(dict(target=family,key=key,gain=gain,validation_mae=sa.macro(rr)))
                print(args.fold,key,m.gp.optimization_,m.gp.diagnostics_,flush=True)
    assert boundary<1e-8 and zero_error<1e-8 and len(trials)==20
    selected={t:min([r for r in trials if r['target']==t],key=lambda r:(r['validation_mae'],r['gain'],r['key'])) for t in protocol['families']}
    joblib.dump(predictions,out/'validation_branch_predictions.joblib',compress=3)
    sa.write_json(out/'result.json',dict(status='VALIDATION_COMPLETE_AUDIT_PENDING',selected=selected,rows=rows,trials=trials,
        models=models,boundary=boundary,zero_reference_error=zero_error,prediction_sha256=sa.digest(out/'validation_branch_predictions.joblib')))
    print(selected,flush=True)


if __name__=='__main__':main()
