"""Shared GP plus source-domain and source-cell random slopes.

For a new cell in a new domain, independent private prior variance is the sum
of domain/cell variances. IDs only construct source covariance, never gates.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from difference_gp import DifferenceGP,cell_contrasts
from mixed_effects_gp import slope_covariance


class HierarchicalSlopesGP(DifferenceGP):
    def __init__(self,kernel=None,optimize=True,rhos=(.1,.1),learn_rhos=True):
        super().__init__(kernel,optimize)
        self.rhos=tuple(rhos);self.learn_rhos=learn_rhos
        if len(self.rhos)!=2 or min(self.rhos)<0 or (learn_rhos and min(self.rhos)<=0):
            raise ValueError('Invalid cell/domain random variances')

    def unpack(self,theta):
        return (theta[:-2],np.exp(theta[-2:])) if self.learn_rhos else (theta,np.asarray(self.rhos))

    def objective(self,theta,gradient=True):
        kt,rhos=self.unpack(theta);kernel=self.kernel.clone_with_theta(kt)
        if gradient:k,dk=kernel(self.X_train_,eval_gradient=True)
        else:k=kernel(self.X_train_)
        c=self.H_@k@self.H_.T+sum(r*s for r,s in zip(rhos,self.nuisance_contrasts_))
        factor=cho_factor(c,lower=True);alpha=cho_solve(factor,self.y_contrast_);n=len(alpha)
        loss=.5*self.y_contrast_@alpha+np.log(np.diag(factor[0])).sum()+.5*n*np.log(2*np.pi)
        if not gradient:return loss
        weight=cho_solve(factor,np.eye(n))-np.outer(alpha,alpha)
        deriv=[.5*np.sum(weight*(self.H_@dk[:,:,j]@self.H_.T)) for j in range(dk.shape[-1])]
        if self.learn_rhos:deriv.extend(.5*np.sum(weight*r*s) for r,s in zip(rhos,self.nuisance_contrasts_))
        return loss,np.asarray(deriv)

    def fit(self,x,y,cell_ids,domain_ids):
        self.X_train_=np.asarray(x,float).copy();y=np.asarray(y,float)
        if y.ndim!=1 or len(y)!=len(x) or len(domain_ids)!=len(y) or not np.isfinite(y).all():raise ValueError('Invalid observations')
        domain_ids=np.asarray(domain_ids);cell_ids=np.asarray(cell_ids)
        if len(cell_ids)!=len(y):raise ValueError('Invalid cell IDs')
        if any(len(set(domain_ids[cell_ids==c]))!=1 for c in set(cell_ids)):raise ValueError('Cell crosses domains')
        self.nuisance_=[slope_covariance(x,ids) for ids in (cell_ids,domain_ids)]
        self.H_=cell_contrasts(cell_ids);self.nuisance_contrasts_=[self.H_@s@self.H_.T for s in self.nuisance_]
        contrast=self.H_@y;self.y_scale_=max(float(np.sqrt(np.mean(contrast**2))),1e-8);self.y_contrast_=contrast/self.y_scale_
        theta=self.kernel.theta;bounds=self.kernel.bounds
        if self.learn_rhos:
            theta=np.r_[theta,np.log(self.rhos)];bounds=np.vstack([bounds,np.tile(np.log([1e-6,1e3]),(2,1))])
        self.optimization_=None
        if self.optimize:
            result=minimize(self.objective,theta,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':100})
            theta=result.x;self.optimization_=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
        kt,rhos=self.unpack(theta);self.rho_cell_,self.rho_domain_=map(float,rhos)
        self.rho_=float(sum(rhos));self.theta_=theta;self.kernel_=self.kernel.clone_with_theta(kt)
        cov=self.H_@self.kernel_(x)@self.H_.T+sum(r*s for r,s in zip(rhos,self.nuisance_contrasts_))
        self.factor_=cho_factor(cov,lower=True);self.alpha_=cho_solve(self.factor_,self.y_contrast_)
        self.point_alpha_=self.H_.T@self.alpha_
        self.diagnostics_=dict(rho_cell=self.rho_cell_,rho_domain=self.rho_domain_,target_rho=self.rho_,
            boundary_hits=[bool(r<=1.0001e-6 or r>=999.9) for r in rhos] if self.learn_rhos else [False,False],
            source_cells=len(set(cell_ids)),source_domains=len(set(domain_ids)))
        return self
