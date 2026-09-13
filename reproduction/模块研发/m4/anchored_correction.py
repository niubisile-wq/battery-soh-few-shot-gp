"""Reference-zero linear/RBF residual functions; source-only weighted fitting."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from correction import features
from source_oof import sa
from features import hi


def reference_features(cell,n,view):
    anchor=float(np.mean(cell.y[:sa.K]));state=np.tile([anchor,anchor,0.,0.],(n,1))
    if view=='state':return state
    ref=hi(cell.x[:sa.K]).mean(0)
    return np.c_[state,np.zeros((n,len(ref))),np.broadcast_to(ref,(n,len(ref))) ]


def difference_kernel(z,z0,x,x0,gamma):
    return rbf_kernel(z,x,gamma=gamma)-rbf_kernel(z0,x,gamma=gamma)-rbf_kernel(z,x0,gamma=gamma)+rbf_kernel(z0,x0,gamma=gamma)


class AnchoredCorrector:
    def __init__(self,view='state',learner='constant',alpha=1.,clip=.05):
        self.view,self.learner,self.alpha,self.clip=view,learner,alpha,clip

    def fit(self,episodes,byid,group):
        self.source_keys=[(e['cell_id'],int(j)) for e in episodes for j in e['query_indices']]
        if self.learner=='constant':return self
        domains={e['domain'] for e in episodes};counts={d:sum(e['domain']==d for e in episodes) for d in domains}
        w=np.concatenate([np.full(len(e['y']),1/len(domains)/counts[e['domain']]/len(e['y'])) for e in episodes])
        y=np.concatenate([e['y']-e['predictions'][group] for e in episodes])
        x=np.concatenate([features(byid[e['cell_id']],e['predictions'][group],e['query_indices'],self.view) for e in episodes])
        x0=np.concatenate([reference_features(byid[e['cell_id']],len(e['y']),self.view) for e in episodes])
        self.keep=np.flatnonzero(x.std(0)>1e-8);assert len(self.keep)
        self.scaler=StandardScaler().fit(x[:,self.keep],sample_weight=w)
        z=self.scaler.transform(x[:,self.keep]);z0=self.scaler.transform(x0[:,self.keep])
        if self.learner=='ridge':self.model=Ridge(alpha=self.alpha,fit_intercept=False).fit(z-z0,y,sample_weight=w*len(w))
        elif self.learner=='rbf':
            self.z,self.z0,self.gamma=z,z0,1/z.shape[1]
            k=difference_kernel(z,z0,z,z0,self.gamma);k=(k+k.T)/2
            sw=np.sqrt(w*len(w));system=sw[:,None]*k*sw[None,:]+np.eye(len(w))*self.alpha
            self.dual=sw*cho_solve(cho_factor(system,lower=True),sw*y)
        else:raise ValueError(self.learner)
        return self

    def predict(self,cell,pred,indices=None):
        if self.learner=='constant':return np.zeros(len(pred))
        if indices is None:indices=np.arange(sa.K,len(cell.x))
        c=sa.inference_view(cell);x=features(c,pred,indices,self.view);x0=reference_features(c,len(pred),self.view)
        z=self.scaler.transform(x[:,self.keep]);z0=self.scaler.transform(x0[:,self.keep])
        r=self.model.predict(z-z0) if self.learner=='ridge' else difference_kernel(z,z0,self.z,self.z0,self.gamma)@self.dual
        return np.clip(r,-self.clip,self.clip)
