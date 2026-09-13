"""Within-cell random slopes plus shared contrast GP; shared-only prediction."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from difference_gp import DifferenceGP,cell_contrasts


def slope_covariance(x,ids):
    x=np.asarray(x,dtype=float);ids=np.asarray(ids)
    if x.ndim!=2 or x.shape[1]==0 or len(x)!=len(ids) or not np.isfinite(x).all():
        raise ValueError('Invalid random slope inputs')
    return (x@x.T/x.shape[1])*(ids[:,None]==ids[None,:])


class MixedEffectsGP(DifferenceGP):
    def __init__(self,kernel=None,optimize=True,rho=.1,learn_rho=True):
        super().__init__(kernel,optimize)
        if rho<0 or (learn_rho and rho<=0):raise ValueError('Invalid rho')
        self.rho,self.learn_rho=rho,learn_rho

    def unpack(self,theta):
        return (theta[:-1],float(np.exp(theta[-1]))) if self.learn_rho else (theta,float(self.rho))

    def objective(self,theta,gradient=True):
        kt,rho=self.unpack(theta);kernel=self.kernel.clone_with_theta(kt)
        if gradient:k,dk=kernel(self.X_train_,eval_gradient=True)
        else:k=kernel(self.X_train_)
        c=self.H_@k@self.H_.T+rho*self.slope_contrast_
        cf=cho_factor(c,lower=True);alpha=cho_solve(cf,self.y_contrast_);n=len(alpha)
        loss=.5*self.y_contrast_@alpha+np.log(np.diag(cf[0])).sum()+.5*n*np.log(2*np.pi)
        if not gradient:return loss
        weight=cho_solve(cf,np.eye(n))-np.outer(alpha,alpha)
        deriv=[.5*np.sum(weight*(self.H_@dk[:,:,j]@self.H_.T)) for j in range(dk.shape[-1])]
        if self.learn_rho:deriv.append(.5*np.sum(weight*rho*self.slope_contrast_))
        return loss,np.asarray(deriv)

    def fit(self,x,y,ids):
        self.X_train_=np.asarray(x,dtype=float).copy();y=np.asarray(y,dtype=float)
        if y.ndim!=1 or len(y)!=len(x) or not np.isfinite(y).all():raise ValueError('Invalid observations')
        self.slope_=slope_covariance(self.X_train_,ids)
        self.H_=cell_contrasts(ids);self.slope_contrast_=self.H_@self.slope_@self.H_.T
        contrast=self.H_@y;self.y_scale_=max(float(np.sqrt(np.mean(contrast**2))),1e-8)
        self.y_contrast_=contrast/self.y_scale_
        theta=self.kernel.theta;bounds=self.kernel.bounds
        if self.learn_rho:
            theta=np.r_[theta,np.log(self.rho)];bounds=np.vstack([bounds,np.log([1e-6,1e3])])
        self.optimization_=None
        if self.optimize:
            result=minimize(self.objective,theta,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':100})
            theta=result.x
            self.optimization_=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
        kt,self.rho_=self.unpack(theta);self.theta_=theta;self.kernel_=self.kernel.clone_with_theta(kt)
        c=self.H_@self.kernel_(self.X_train_)@self.H_.T+self.rho_*self.slope_contrast_
        self.factor_=cho_factor(c,lower=True);self.alpha_=cho_solve(self.factor_,self.y_contrast_)
        self.point_alpha_=self.H_.T@self.alpha_
        self.diagnostics_=dict(rho=self.rho_,rho_bound_hit=bool(self.learn_rho and (self.rho_<=1.0001e-6 or self.rho_>=999.9)),
            shared_kernel=str(self.kernel_),random_contrast_trace=float(self.rho_*np.trace(self.slope_contrast_)))
        return self

    # Inherited prediction uses only shared kernel cross-covariance; never source IDs.
