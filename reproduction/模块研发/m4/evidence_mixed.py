"""Support-contrast Gaussian evidence weights for two anchored adapters.

Equal prior probabilities; evidence from the permitted first10 labels only.
This is not a calibrated reliability score or full posterior over model selection.
"""
import numpy as np
from types import SimpleNamespace
from scipy.linalg import helmert,cho_factor,cho_solve
from scipy.special import expit
from conditional_mixed import ConditionalMixed
from source_oof import sa


class EvidenceMixed:
    def __init__(self,parent,mode='evidence'):
        assert mode in ('evidence','uniform')
        self.parent,self.mode=parent,mode;self.source_keys=parent.source_keys

    def weights(self,c):
        early=SimpleNamespace(x=c.x[:sa.K],reference_capacity=c.reference_capacity)
        gp=self.parent.gp;z=ConditionalMixed(self.parent).latent(early);h=helmert(sa.K,full=False)
        cross=gp.kernel_(z,gp.X_train_)@gp.H_.T
        shared=gp.kernel_(z)-cross@cho_solve(gp.factor_,cross.T)
        slope=gp.rho_*(z@z.T)/z.shape[1]
        y=np.asarray(c.y[:sa.K],float);assert np.isfinite(y).all()
        residual=h@(y/gp.y_scale_-gp.predict(z)/gp.y_scale_)
        log_evidence=[]
        for cov in (shared,shared+slope):
            s=h@cov@h.T;s=(s+s.T)/2;factor=cho_factor(s,lower=True)
            log_evidence.append(float(-.5*residual@cho_solve(factor,residual)-np.log(np.diag(factor[0])).sum()
                -.5*len(residual)*np.log(2*np.pi)))
        private=float(expit(log_evidence[1]-log_evidence[0])) if self.mode=='evidence' else .5
        return dict(private=private,shared=1-private,log_evidence_shared=log_evidence[0],log_evidence_private=log_evidence[1])

    def predict(self,c):
        weight=self.weights(c)['private']
        private=ConditionalMixed(self.parent,True).predict(c)
        shared=ConditionalMixed(self.parent,False).predict(c)
        return (1-weight)*shared+weight*private
