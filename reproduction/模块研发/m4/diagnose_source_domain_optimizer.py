"""Read-only gradient/conditioning checks at saved source-domain fit endpoints."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE


def main():
    root=HERE.parent/'results/m4_source_domain_pilot_v1';r=json.loads((root/'result.json').read_text())
    reports=[]
    with threadpool_limits(limits=1):
        for spec in r['models']:
            path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];gp=joblib.load(path).gp
            theta=gp.theta_.copy();bounds=np.vstack([gp.kernel.bounds,np.log([1e-6,1e3])])
            loss,gradient=gp.objective(theta)
            projected=gradient.copy()
            for i,(lo,hi) in enumerate(bounds):
                if theta[i]<=lo+1e-8 and gradient[i]>0:projected[i]=0
                if theta[i]>=hi-1e-8 and gradient[i]<0:projected[i]=0
            differences=[]
            for eps in (1e-3,1e-4,1e-5):
                numeric=[]
                for i,(lo,hi) in enumerate(bounds):
                    left=theta.copy();right=theta.copy()
                    left[i]=max(lo,theta[i]-eps);right[i]=min(hi,theta[i]+eps)
                    numeric.append(float((gp.objective(right,False)-gp.objective(left,False))/(right[i]-left[i])))
                differences.append(dict(step=eps,gradient=numeric,max_abs_error=float(np.max(abs(np.array(numeric)-gradient)))))
            covariance=gp.H_@gp.kernel_(gp.X_train_)@gp.H_.T+gp.rho_*gp.slope_contrast_
            eigen=np.linalg.eigvalsh((covariance+covariance.T)/2)
            report=dict(kernel=spec['kernel'],optimization=gp.optimization_,theta=theta.tolist(),bounds=bounds.tolist(),
                objective=float(loss),gradient=gradient.tolist(),projected_gradient_inf=float(np.max(abs(projected))),
                finite_difference=differences,min_covariance_eigenvalue=float(eigen[0]),
                covariance_condition=float(eigen[-1]/eigen[0]),model_sha256=spec['sha256'])
            reports.append(report);print(report,flush=True)
            assert sa.digest(path)==spec['sha256']
    out=HERE.parent/'results/m4_source_domain_optimizer_diagnosis_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'summary.json',dict(status='SAVED_ENDPOINT_DIAGNOSIS',reports=reports,
        source_sha256=sa.digest(root/'result.json'),code_sha256=sa.digest(Path(__file__)),
        limits='No optimization restart or parameter change. Bound coordinate differences can be one-sided; diagnostics do not certify global optimality or explain all line-search failures.'))


if __name__=='__main__':main()
