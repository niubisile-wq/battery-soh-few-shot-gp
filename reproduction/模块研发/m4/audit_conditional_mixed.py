"""Independent adjacent-support contrasts (correlated noise) and validation audit."""
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from multiscale import local_features
from score import PARENTS


def independent_predict(model,c,family,point_mean=False):
    gp=model.gp;z=model.scaler.transform(model.pca.transform(model.raw_scaler.transform(
        local_features(c)[:,model.keep]))[:,model.latent_keep])
    if family=='bias':
        mean=gp.kernel_(z,gp.X_train_)@gp.point_alpha_*gp.y_scale_
        return mean[sa.K:]+float(np.mean(c.y[:sa.K]))-float(np.mean(mean[:sa.K]))
    zs=z[:sa.K];d=np.diff(np.eye(sa.K),axis=0);rho=gp.rho_ if family=='private' else 0.
    cross_s=gp.kernel_(zs,gp.X_train_)@gp.H_.T
    weights_s=cho_solve(gp.factor_,cross_s.T)
    cov_s=gp.kernel_(zs)+rho*(zs@zs.T)/z.shape[1]-cross_s@weights_s
    # Optional fixed source-mean evaluation order. Target covariance remains
    # independently evaluated with adjacent contrasts and correlated noise.
    point_alpha=gp.H_.T@gp.alpha_ if point_mean else None
    mu_s=gp.kernel_(zs,gp.X_train_)@point_alpha if point_mean else cross_s@gp.alpha_
    coeff=np.linalg.solve(d@cov_s@d.T,d@(np.asarray(c.y[:sa.K],float)/gp.y_scale_-mu_s))
    output=[]
    for j in range(0,len(z),256):
        zz=z[j:j+256];cross=gp.kernel_(zz,gp.X_train_)@gp.H_.T
        cov=gp.kernel_(zz,zs)+rho*(zz@zs.T)/z.shape[1]-cross@weights_s
        mean=gp.kernel_(zz,gp.X_train_)@point_alpha if point_mean else cross@gp.alpha_
        output.append((mean+cov@d.T@coeff)*gp.y_scale_)
    pred=np.concatenate(output)
    return pred[sa.K:]+np.mean(np.asarray(c.y[:sa.K],float))-pred[:sa.K].mean()


def main():
    root=HERE.parent/'results/m4_conditional_mixed_validation_v1'
    req=json.loads((root/'request.json').read_text());allresults=json.loads((root/'result.json').read_text())
    assert allresults['status']=='ALL_VALIDATION_SELECTED_AUDIT_PENDING' and len(allresults['selections'])==21
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    byds={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')}
    prediction_error=0.;row_error=0.;trial_error=0.;total_rows=0
    with threadpool_limits(limits=1):
        for selection in allresults['selections']:
            folder=Path(selection['root']);assert sa.digest(folder/'result.json')==selection['result_sha256']
            result=json.loads((folder/'result.json').read_text());fold=result['fold']
            assert sa.digest(Path(result['source_root'])/'result.json')==result['source_result_sha256']
            assert sa.digest(Path(result['parent_path']))==result['parent_sha256']
            assert sa.digest(folder/'predictions.joblib')==result['prediction_sha256']
            vp=joblib.load(result['parent_path']);saved=joblib.load(folder/'predictions.joblib')
            _,val,_=sa.split(byds[fold.split(':')[0]],fold);byid={c.id:c for c in val};pred={}
            for spec in result['models']:
                assert sa.digest(Path(spec['path']))==spec['sha256'];model=joblib.load(spec['path'])
                kernel=spec['key'].split('__')[1]
                for family in req['protocol']['families']:
                    key=family+'__'+kernel
                    for c in val:
                        p=independent_predict(model,sa.inference_view(c),family);pred[c.id,key]=p
                        prediction_error=max(prediction_error,float(np.max(abs(p-saved[c.id,key]))))
            assert set(pred)==set(saved)
            keys={f+'__'+k for f in req['protocol']['families'] for k in req['protocol']['kernels']}
            expected={(cid,k,g,a) for cid in byid for k in keys for g in PARENTS for a in req['protocol']['gains']}
            assert len(result['rows'])==len(expected)
            assert {(r['cell_id'],r['key'],r['group'],r['gain']) for r in result['rows']}==expected
            values={}
            for row in result['rows']:
                cid,key,g,a=row['cell_id'],row['key'],row['group'],row['gain'];c=byid[cid]
                loss=float(np.mean(abs((1-a)*vp[cid,g]+a*pred[cid,key]-c.y[sa.K:])))
                row_error=max(row_error,abs(loss-row['mae']))
                if g=='B123':values.setdefault((key,a),{}).setdefault(c.domain,[]).append(loss)
            assert len(result['trials'])==30 and {(t['key'],t['gain']) for t in result['trials']}==set(values)
            trials=[]
            for t in result['trials']:
                loss=float(np.mean([np.mean(v) for v in values[t['key'],t['gain']].values()]))
                trial_error=max(trial_error,abs(loss-t['validation_mae']));trials.append(dict(t,validation_mae=loss))
            for family in req['protocol']['families']:
                best=min([t for t in trials if t['target']==family],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
                chosen=result['selected'][family]
                assert best['key']==chosen['key'] and best['gain']==chosen['gain']
            assert result['selected']==selection['selected'];total_rows+=len(expected)
            print(fold,'independent adjacent-contrast audit',flush=True)
    assert max(prediction_error,row_error,trial_error)<1e-8
    sa.write_json(root/'verification.json',dict(status='PASS',folds=21,trials=630,rows=total_rows,
        prediction_error=prediction_error,row_error=row_error,trial_error=trial_error,
        result_sha256=sa.digest(root/'result.json'),request_sha256=sa.digest(root/'request.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Independent target contrast basis with correlated noise; uses audited saved source factors, no new GP/encoder fitting or held-domain confirmation.'))
    print('PASS21folds630objectives',flush=True)


if __name__=='__main__':main()
