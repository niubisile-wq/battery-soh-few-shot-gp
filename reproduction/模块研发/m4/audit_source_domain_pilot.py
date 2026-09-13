"""Fresh inner encoder/GP refits and independent sparse held-source predictions."""
import json,argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from difference_branch import DifferenceBranch
from mixed_effects_branch import MixedEffectsBranch
from support_cv_mixed import SupportCVMixed
from audit_support_cv import independent_weight
from audit_conditional_mixed import independent_predict
from score import PARENTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(HERE.parent/'results/m4_source_domain_pilot_v1'));args=ap.parse_args()
    root=Path(args.root).resolve()
    req=json.loads((root/'request.json').read_text());result=json.loads((root/'result.json').read_text())
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    source=Path(req['origin']['root'])
    for name,key in [('request.json','request_sha256'),('audit_manifest.json','manifest_sha256'),
                     ('verification.json','audit_sha256'),('complete_parent_source_oof.joblib','cache_sha256')]:
        assert sa.digest(source/name)==req['origin'][key]
    train,_,_=sa.split(sa.load_cells(req['fold'].split(':')[0]),req['fold'])
    original=json.loads((source/'request.json').read_text());allkeys=set(map(tuple,original['source_keys']))
    byid={c.id:c for c in train};fitkeys={k for k in allkeys if byid[k[0]].domain!=req['held']}
    assert fitkeys==set(map(tuple,req['fit_keys']))
    allowed=defaultdict(list)
    for cid,j in sorted(fitkeys):allowed[cid].append(j)
    fit=[sa.source_view(c,allowed[c.id]) for c in train if c.domain!=req['held']]
    held={c.id:c for c in train if c.domain==req['held']};assert set(allowed).isdisjoint(held)
    episodes={e['cell_id']:e for e in joblib.load(source/'complete_parent_source_oof.joblib') if e['domain']==req['held']}
    assert set(episodes)==set(held)
    assert sa.digest(root/'predictions.joblib')==result['prediction_sha256'];saved=joblib.load(root/'predictions.joblib')
    errors=dict(encoder=0.,theta=0.,refit_prediction=0.,independent_prediction=0.,row=0.)
    pred={};optimizers=[]
    with threadpool_limits(limits=1):
        for spec in result['models']:
            path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];old=joblib.load(path)
            ref=DifferenceBranch('difference',spec['kernel'],optimize=False).fit(fit,dict(allowed))
            fresh=MixedEffectsBranch.from_encoder(ref,fit,dict(allowed),optimize=True)
            assert old.source_keys==fresh.source_keys and set(fresh.source_keys)==fitkeys
            for attr in ('keep','latent_keep'):np.testing.assert_array_equal(getattr(old,attr),getattr(fresh,attr))
            for scaler in ('raw_scaler','scaler'):
                for attr in ('mean_','scale_'):
                    errors['encoder']=max(errors['encoder'],float(np.max(abs(getattr(getattr(old,scaler),attr)-getattr(getattr(fresh,scaler),attr)))))
            errors['encoder']=max(errors['encoder'],float(np.max(abs(old.gp.X_train_-fresh.gp.X_train_))))
            errors['theta']=max(errors['theta'],float(np.max(abs(old.gp.theta_-fresh.gp.theta_))))
            assert fresh.gp.optimization_==old.gp.optimization_==spec['optimization']
            optimizers.append(dict(kernel=spec['kernel'],optimization=fresh.gp.optimization_,diagnostics=fresh.gp.diagnostics_))
            for cid,e in episodes.items():
                c=held[cid];q=np.asarray(e['query_indices'],int)
                assert {(cid,int(j)) for j in q}=={k for k in allkeys if k[0]==cid and k[1]>=sa.K}
                assert np.array_equal(e['y'],c.y[q]);ix=np.r_[np.arange(sa.K),q]
                sparse=replace(c,x=c.x[ix],y=c.y[ix],cycle=c.cycle[ix]);hidden=sa.inference_view(sparse)
                w=independent_weight(old,hidden)['private']
                shared=independent_predict(old,hidden,'shared',point_mean=True)
                private=independent_predict(old,hidden,'private',point_mean=True)
                for family in req['protocol']['families']:
                    a=w if family=='cv' else .5;key=family+'__'+spec['kernel']
                    p=shared+a*(private-shared);pred[cid,key]=p
                    errors['independent_prediction']=max(errors['independent_prediction'],float(np.max(abs(p-saved[cid,key]))))
                    freshp=SupportCVMixed(fresh,family).predict(hidden)
                    errors['refit_prediction']=max(errors['refit_prediction'],float(np.max(abs(freshp-saved[cid,key]))))
            print(spec['kernel'],'fresh inner encoder/GP audit',fresh.gp.optimization_,flush=True)
    assert set(pred)==set(saved)
    keys={f+'__'+k for f in req['protocol']['families'] for k in req['protocol']['kernels']}
    expected={(cid,k,g,a) for cid in held for k in keys for g in PARENTS for a in req['protocol']['gains']}
    assert len(result['rows'])==len(expected) and {(r['cell_id'],r['key'],r['group'],r['gain']) for r in result['rows']}==expected
    for r in result['rows']:
        cid,key,g,a=r['cell_id'],r['key'],r['group'],r['gain'];e=episodes[cid]
        assert r['domain']==req['held']
        p=(1-a)*e['predictions'][g]+a*pred[cid,key]
        errors['row']=max(errors['row'],abs(float(np.mean(abs(p-e['y'])))-r['mae']))
    assert max(errors.values())<1e-8,errors
    failures=[r for r in optimizers if not r['optimization']['success']]
    sa.write_json(root/'verification.json',dict(status='REPLAY_PASS_OPTIMIZER_FAILURE_PRESENT' if failures else 'PASS',
        errors=errors,rows=len(expected),optimized_refits=2,nonoptimized_reference_refits=2,optimizers=optimizers,
        result_sha256=sa.digest(root/'result.json'),request_sha256=sa.digest(root/'request.json'),code_sha256=sa.digest(Path(__file__)),
        helper_hashes={n:sa.digest(HERE/n) for n in ('audit_support_cv.py','audit_conditional_mixed.py')},
        limits='Fresh fits use original implementation; independent target arithmetic uses alternative contrast basis. Reproducibility does not turn optimizer failure into convergence.'))
    print('replay passed',errors,'optimizer failures',len(failures),flush=True)


if __name__=='__main__':main()
