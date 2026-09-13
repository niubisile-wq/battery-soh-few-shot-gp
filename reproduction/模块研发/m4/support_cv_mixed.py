"""First-ten inner-support predictive mixing, with fixed noise-scale shrinkage.

Inner training uses labels0:t and predicts t, t=5..9. All first-ten input
curves define the fixed feature reference: this is not an online forecast
benchmark. Later curves/labels never enter mixing weights. No new source fit.
"""
from types import SimpleNamespace
import numpy as np
from scipy.linalg import helmert,cho_solve
from conditional_mixed import ConditionalMixed
from source_oof import sa


def prefix_prediction(gp,z,y,t,private):
    zs=z[:t];zz=z[t:t+1];h=helmert(t,full=False)
    rho=gp.rho_ if private else 0.;d=z.shape[1]
    ks=gp.kernel_(zs,gp.X_train_)@gp.H_.T
    solved=cho_solve(gp.factor_,ks.T)
    observed=gp.kernel_(zs)-ks@solved+rho*(zs@zs.T)/d
    latent=gp.kernel_(zs,zs)-ks@solved+rho*(zs@zs.T)/d
    cov=h@observed@h.T;cov=(cov+cov.T)/2
    mean=gp.predict(zs)/gp.y_scale_
    a=np.linalg.solve(cov,h@(y[:t]/gp.y_scale_-mean))
    updated=mean+latent@h.T@a
    anchor=np.mean(y[:t])/gp.y_scale_-np.mean(updated)
    kq=gp.kernel_(zz,gp.X_train_)@gp.H_.T
    cross=gp.kernel_(zz,zs)-kq@solved+rho*(zz@zs.T)/d
    return float((gp.predict(zz)/gp.y_scale_+cross@h.T@a+anchor)[0]*gp.y_scale_)


class SupportCVMixed:
    def __init__(self,parent,mode='cv'):
        assert mode in ('cv','uniform')
        self.parent,self.mode=parent,mode;self.source_keys=parent.source_keys

    def weights(self,c):
        early=SimpleNamespace(x=c.x[:sa.K],reference_capacity=c.reference_capacity)
        z=ConditionalMixed(self.parent).latent(early);y=np.asarray(c.y[:sa.K],float)
        assert len(y)==10 and np.isfinite(y).all()
        gp=self.parent.gp
        shared=np.array([prefix_prediction(gp,z,y,t,False) for t in range(5,10)])
        private=np.array([prefix_prediction(gp,z,y,t,True) for t in range(5,10)])
        delta=private-shared;residual=y[5:10]-shared
        penalty=5*float(gp.kernel_.k2.noise_level)*gp.y_scale_**2
        assert penalty>0
        w=float(np.clip((delta@residual+.5*penalty)/(delta@delta+penalty),0,1)) if self.mode=='cv' else .5
        return dict(private=w,shared=1-w,penalty=penalty,
                    inner_shared_mse=float(np.mean(residual**2)),
                    inner_private_mse=float(np.mean((y[5:10]-private)**2)))

    def predict(self,c):
        w=self.weights(c)['private']
        return (1-w)*ConditionalMixed(self.parent,False).predict(c)+w*ConditionalMixed(self.parent,True).predict(c)
