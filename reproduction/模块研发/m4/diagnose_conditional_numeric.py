"""Diagnose basis/association roundoff at one independently localized mismatch."""
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import cho_solve,helmert
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from conditional_mixed import ConditionalMixed
from audit_conditional_mixed import independent_predict


def main():
    root=HERE.parent/'results/m4_conditional_mixed_screen_v1'
    failure=json.loads((root/'audit_numeric_failure_8e8bca52cade.json').read_text())
    worst=failure['worst_prediction'];fold=worst['fold'];cid=worst['cell_id']
    frozen=json.loads((root/'frozen_selections.json').read_text())
    selected=next(s for s in frozen['selections'] if s['fold']==fold)['choices']['private']
    model=joblib.load(selected['model']['path']);gp=model.gp
    c=sa.inference_view(next(c for c in sa.load_cells('Tongji') if c.id==cid))
    m=ConditionalMixed(model,True);z=m.latent(c);zs=z[:sa.K];rho=gp.rho_
    path=root/'folds'/fold.replace(':','__')/'predictions'/(cid+'.npz')
    with np.load(path) as p:stored=p['branch_private'].copy();y=p['y'].copy()
    rows=[]
    with threadpool_limits(limits=1):
        primary=m.predict(c);independent=independent_predict(model,c,'private')
        cross_s=gp.kernel_(zs,gp.X_train_)@gp.H_.T;solved=cho_solve(gp.factor_,cross_s.T)
        d=np.diff(np.eye(sa.K),axis=0)
        bases={'adjacent':d,'qr_adjacent':np.linalg.qr(d.T,mode='reduced')[0].T,'helmert':helmert(sa.K,full=False)}
        for basis_name,basis in bases.items():
            for association in ('cross_then_alpha','point_alpha'):
                for symmetric in (False,True):
                    # Preserve primary covariance addition order to isolate basis/means.
                    ss=gp.kernel_(zs)-cross_s@solved+rho*(zs@zs.T)/z.shape[1]
                    latent_ss=gp.kernel_(zs,zs)-cross_s@solved+rho*(zs@zs.T)/z.shape[1]
                    css=basis@ss@basis.T
                    if symmetric:css=(css+css.T)/2
                    mu_s=cross_s@gp.alpha_ if association=='cross_then_alpha' else gp.predict(zs)/gp.y_scale_
                    weights=np.linalg.solve(css,basis@(np.asarray(c.y[:sa.K],float)/gp.y_scale_-mu_s))
                    updated_s=mu_s+latent_ss@basis.T@weights
                    anchor=np.asarray(c.y[:sa.K],float).mean()/gp.y_scale_-updated_s.mean()
                    outputs=[]
                    for j in range(sa.K,len(z),256):
                        zz=z[j:j+256];cross=gp.kernel_(zz,gp.X_train_)@gp.H_.T
                        cov=gp.kernel_(zz,zs)-cross@solved+rho*(zz@zs.T)/z.shape[1]
                        mu=cross@gp.alpha_ if association=='cross_then_alpha' else gp.predict(zz)/gp.y_scale_
                        outputs.append((mu+cov@basis.T@weights+anchor)*gp.y_scale_)
                    p=np.concatenate(outputs)
                    rows.append(dict(basis=basis_name,association=association,symmetric=symmetric,
                        condition=float(np.linalg.cond(css)),max_prediction_error=float(np.max(abs(p-stored))),
                        rmse_delta_pp=float((np.sqrt(np.mean((p-y)**2))-np.sqrt(np.mean((stored-y)**2)))*100)))
    out=HERE.parent/'results/m4_conditional_numeric_diagnosis_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'summary.json',dict(status='NUMERIC_DIAGNOSIS_ONLY',fold=fold,cell_id=cid,
        primary_replay_error=float(np.max(abs(primary-stored))),independent_error=float(np.max(abs(independent-stored))),
        models_unmodified=True,source_model_sha256=sa.digest(Path(selected['model']['path'])),variants=rows,
        code_sha256=sa.digest(Path(__file__)),limits='One localized case; basis and arithmetic variants diagnose floating point, not model candidates or changed acceptance tolerance.'))
    print('primary',float(np.max(abs(primary-stored))),'independent',float(np.max(abs(independent-stored))),flush=True)
    for r in rows:print(r,flush=True)


if __name__=='__main__':main()
