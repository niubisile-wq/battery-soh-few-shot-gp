"""Label-free first10 information about target private slopes, all cells/kernels."""
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import cho_solve,helmert
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from conditional_mixed import ConditionalMixed
from evaluate import dump_csv


def information(model,c):
    gp=model.gp;z=ConditionalMixed(model,True).latent(c);zs=z[:sa.K];h=helmert(sa.K,full=False)
    cross=gp.kernel_(zs,gp.X_train_)@gp.H_.T
    ss=gp.kernel_(zs)-cross@cho_solve(gp.factor_,cross.T)
    noise=h@ss@h.T;noise=(noise+noise.T)/2
    design=h@zs;prior=gp.rho_/z.shape[1]
    fisher=prior*design.T@np.linalg.solve(noise,design);fisher=(fisher+fisher.T)/2
    eigen=np.linalg.eigvalsh(fisher)
    assert eigen.min()>-1e-6 and prior>0
    posterior_fraction=np.linalg.solve(np.eye(z.shape[1])+fisher,np.eye(z.shape[1]))
    delta=z[sa.K:]-zs.mean(0)
    prior_var=np.sum(delta**2,axis=1)
    post_var=np.einsum('ij,jk,ik->i',delta,posterior_fraction,delta)
    ratio=np.divide(post_var,prior_var,out=np.ones_like(post_var),where=prior_var>1e-12)
    assert ratio.min()>-1e-6 and ratio.max()<1+1e-6
    return dict(latent_dim=z.shape[1],rho=gp.rho_,
        support_design_rank=int(np.linalg.matrix_rank(design)),
        information_eigenvalues=eigen.tolist(),
        effective_learned_directions=float(np.sum(eigen/(1+eigen))),
        strongest_direction_information=float(eigen.max()),
        private_variance_trace_remaining=float(np.trace(posterior_fraction)/z.shape[1]),
        query_direction_variance_remaining_mean=float(ratio.mean()),
        query_direction_variance_remaining_p95=float(np.percentile(ratio,95)))


def main():
    results=HERE.parent/'results';out=results/'m4_support_information_v1';out.mkdir(exist_ok=False)
    batch=json.loads((results/'m4_mixed_effects_batch_v1/result.json').read_text())
    assert batch['status']=='VALIDATION_BATCH_COMPLETE'
    byds={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')};rows=[]
    with threadpool_limits(limits=1):
        for fold in batch['folds']:
            root=Path(fold['branch']);r=json.loads((root/'result.json').read_text());audit=json.loads((root/'verification.json').read_text())
            assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
            _,_,test=sa.split(byds[fold['fold'].split(':')[0]],fold['fold'])
            for spec in r['models']:
                if not spec['key'].startswith('mixed__'):continue
                assert sa.digest(Path(spec['path']))==spec['sha256'];m=joblib.load(spec['path'])
                for c in test:
                    # Explicitly mask ALL labels, including support: geometry-only.
                    from dataclasses import replace
                    masked=replace(c,y=np.full(len(c.y),np.nan))
                    rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,kernel=spec['key'],
                        model_sha256=spec['sha256'],**information(m,masked)))
            print(fold['fold'],'label-free information checked',flush=True)
    assert len(rows)==730
    summaries=[]
    fields=['effective_learned_directions','private_variance_trace_remaining','query_direction_variance_remaining_mean']
    for ds in byds:
        for kernel in ('mixed__rbf','mixed__matern'):
            selected=[r for r in rows if r['dataset']==ds and r['kernel']==kernel]
            domains=sorted({r['domain'] for r in selected})
            summaries.append(dict(dataset=ds,kernel=kernel,**{k:float(np.mean([
                np.mean([r[k] for r in selected if r['domain']==d]) for d in domains])) for k in fields}))
    dump_csv(out/'cells.csv',[dict(r,information_eigenvalues=json.dumps(r['information_eigenvalues'])) for r in rows])
    sa.write_json(out/'summary.json',dict(status='LABEL_FREE_INFORMATION_DIAGNOSIS',rows=730,table=summaries,
        code_sha256=sa.digest(Path(__file__)),
        limits='No support/query label values used. Model-based posterior variance,not calibrated accuracy or causal proof. No fitted gates or model selection. Uses reused development input geometry.'))
    for r in summaries:print(r,flush=True)


if __name__=='__main__':main()
