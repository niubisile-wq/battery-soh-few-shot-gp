"""Independent Gaussian evidence ratio using adjacent differences; no future labels."""
import json
from pathlib import Path
import numpy as np
import joblib
from scipy.linalg import cho_solve
from scipy.special import expit
from threadpoolctl import threadpool_limits
from types import SimpleNamespace
from source_oof import sa,HERE
from conditional_mixed import ConditionalMixed
from audit_conditional_mixed import independent_predict
from score import PARENTS


def independent_weight(model,c):
    gp=model.gp;z=ConditionalMixed(model).latent(SimpleNamespace(x=c.x[:sa.K],reference_capacity=c.reference_capacity))
    d=np.diff(np.eye(sa.K),axis=0);cross=gp.kernel_(z,gp.X_train_)@gp.H_.T
    shared=gp.kernel_(z)-cross@cho_solve(gp.factor_,cross.T);slope=gp.rho_*(z@z.T)/z.shape[1]
    mean=gp.kernel_(z,gp.X_train_)@(gp.H_.T@gp.alpha_)
    residual=d@(np.asarray(c.y[:sa.K],float)/gp.y_scale_-mean);logs=[]
    for cov in (shared,shared+slope):
        s=d@cov@d.T;s=(s+s.T)/2;sign,logdet=np.linalg.slogdet(s);assert sign>0
        logs.append(float(-.5*(residual@np.linalg.solve(s,residual)+logdet)))
    # Basis Jacobian and Gaussian constant cancel in the log evidence ratio.
    ratio=logs[1]-logs[0]
    return float(expit(ratio)),ratio


def main():
    root=HERE.parent/'results/m4_evidence_validation_v1';req=json.loads((root/'request.json').read_text())
    allresults=json.loads((root/'result.json').read_text());assert len(allresults['selections'])==21
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    byds={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')}
    errors=dict(weight=0.,log_ratio=0.,prediction=0.,row=0.,trial=0.);total_rows=0
    with threadpool_limits(limits=1):
        for entry in allresults['selections']:
            folder=Path(entry['root']);assert sa.digest(folder/'result.json')==entry['result_sha256']
            result=json.loads((folder/'result.json').read_text());fold=result['fold']
            assert sa.digest(Path(result['source_root'])/'result.json')==result['source_result_sha256']
            assert sa.digest(Path(result['parent_path']))==result['parent_sha256']
            assert sa.digest(folder/'predictions.joblib')==result['prediction_sha256']
            saved=joblib.load(folder/'predictions.joblib');vp=joblib.load(result['parent_path'])
            _,val,_=sa.split(byds[fold.split(':')[0]],fold);byid={c.id:c for c in val};pred={}
            for spec in result['models']:
                assert sa.digest(Path(spec['path']))==spec['sha256'];m=joblib.load(spec['path']);kernel=spec['key'].split('__')[1]
                for c in val:
                    masked=sa.inference_view(c);weight,ratio=independent_weight(m,masked);oldw=saved['weights'][c.id,kernel]
                    errors['weight']=max(errors['weight'],abs(weight-oldw['private']))
                    errors['log_ratio']=max(errors['log_ratio'],abs(ratio-(oldw['log_evidence_private']-oldw['log_evidence_shared'])))
                    shared=independent_predict(m,masked,'shared',point_mean=True)
                    private=independent_predict(m,masked,'private',point_mean=True)
                    for family in req['protocol']['families']:
                        a=weight if family=='evidence' else .5;key=family+'__'+kernel;p=(1-a)*shared+a*private
                        pred[c.id,key]=p;errors['prediction']=max(errors['prediction'],float(np.max(abs(p-saved['predictions'][c.id,key]))))
            assert set(pred)==set(saved['predictions'])
            keys={f+'__'+k for f in req['protocol']['families'] for k in req['protocol']['kernels']}
            expected={(cid,k,g,a) for cid in byid for k in keys for g in PARENTS for a in req['protocol']['gains']}
            assert len(result['rows'])==len(expected)
            assert {(r['cell_id'],r['key'],r['group'],r['gain']) for r in result['rows']}==expected
            values={}
            for row in result['rows']:
                cid,key,g,a=row['cell_id'],row['key'],row['group'],row['gain'];c=byid[cid]
                loss=float(np.mean(abs((1-a)*vp[cid,g]+a*pred[cid,key]-c.y[sa.K:])))
                errors['row']=max(errors['row'],abs(loss-row['mae']))
                if g=='B123':values.setdefault((key,a),{}).setdefault(c.domain,[]).append(loss)
            trials=[];assert len(result['trials'])==20 and {(t['key'],t['gain']) for t in result['trials']}==set(values)
            for t in result['trials']:
                loss=float(np.mean([np.mean(v) for v in values[t['key'],t['gain']].values()]))
                errors['trial']=max(errors['trial'],abs(loss-t['validation_mae']));trials.append(dict(t,validation_mae=loss))
            for family in req['protocol']['families']:
                best=min([t for t in trials if t['target']==family],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
                assert best['key']==result['selected'][family]['key'] and best['gain']==result['selected'][family]['gain']
            assert result['selected']==entry['selected'];total_rows+=len(expected)
            print(fold,'independent evidence/validation audit',flush=True)
    assert max(errors.values())<1e-8,errors
    sa.write_json(root/'verification.json',dict(status='PASS',errors=errors,rows=total_rows,trials=420,folds=21,
        result_sha256=sa.digest(root/'result.json'),request_sha256=sa.digest(root/'request.json'),code_sha256=sa.digest(Path(__file__)),
        helper_sha256=sa.digest(HERE/'audit_conditional_mixed.py'),
        limits='Independent contrast-basis evidence ratio and target posterior, saved audited source model; no new source fitting or generalization confirmation.'))
    print('PASS420objectives',errors,flush=True)


if __name__=='__main__':main()
