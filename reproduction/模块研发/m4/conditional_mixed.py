"""Target first-K contrast conditioning, shared posterior plus private slopes.

Source nuisance affects source covariance only; new target slopes are independent.
The target intercept is removed with orthonormal support contrasts. After updating,
the predicted support mean is anchored to the observed support mean.
"""
import numpy as np
from scipy.linalg import cho_solve,helmert
from multiscale import local_features
from source_oof import sa


class ConditionalMixed:
    def __init__(self,parent,private=True):
        self.parent,self.private=parent,private
        self.source_keys=parent.source_keys

    def latent(self,c):
        m=self.parent
        return m.scaler.transform(m.pca.transform(m.raw_scaler.transform(
            local_features(c)[:,m.keep]))[:,m.latent_keep])

    def predict(self,c):
        z=self.latent(c);gp=self.parent.gp;zs=z[:sa.K]
        ys=np.asarray(c.y[:sa.K],dtype=float)
        assert len(ys)==sa.K and np.isfinite(ys).all()
        hs=helmert(sa.K,full=False);rho=gp.rho_ if self.private else 0.
        # Cross calls exclude independent measurement white noise.
        support_cross=gp.kernel_(zs,gp.X_train_)@gp.H_.T
        solved=cho_solve(gp.factor_,support_cross.T)
        cov_ss=gp.kernel_(zs)-support_cross@solved+rho*(zs@zs.T)/z.shape[1]
        css=hs@cov_ss@hs.T;css=(css+css.T)/2
        mean_s=gp.predict(zs)/gp.y_scale_
        weights=np.linalg.solve(css,hs@(ys/gp.y_scale_-mean_s))
        # For the latent support mean update, omit observation white noise.
        latent_ss=gp.kernel_(zs,zs)-support_cross@solved+rho*(zs@zs.T)/z.shape[1]
        updated_s=mean_s+latent_ss@hs.T@weights
        anchor=float(ys.mean()/gp.y_scale_-updated_s.mean())
        out=[]
        for j in range(sa.K,len(z),256):
            zz=z[j:j+256];cross=gp.kernel_(zz,gp.X_train_)@gp.H_.T
            cov_qs=gp.kernel_(zz,zs)-cross@solved+rho*(zz@zs.T)/z.shape[1]
            mean_q=gp.predict(zz)/gp.y_scale_
            out.append((mean_q+cov_qs@hs.T@weights+anchor)*gp.y_scale_)
        result=np.concatenate(out)
        assert np.isfinite(result).all()
        return result
