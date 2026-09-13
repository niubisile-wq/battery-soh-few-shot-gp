"""Bounded source-trained conditional residual for a frozen complete parent."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from source_oof import sa
from features import hi


def features(cell,pred,indices,view):
    p=np.asarray(pred,dtype=float);anchor=float(np.mean(cell.y[:sa.K]));d=p-anchor
    state=np.c_[p,np.full(len(p),anchor),d,d*d]
    if view=='state':return state
    if view!='signal':raise ValueError(view)
    h=hi(cell.x);ref=h[:sa.K].mean(0)
    return np.c_[state,h[indices]-ref,np.broadcast_to(ref,(len(p),len(ref))) ]


class ResidualCorrector:
    def __init__(self,view='state',learner='constant',alpha=1.,clip=.05):
        self.view,self.learner,self.alpha,self.clip=view,learner,alpha,clip

    def fit(self,episodes,byid,group):
        domains={e['domain'] for e in episodes};counts={d:sum(e['domain']==d for e in episodes) for d in domains}
        w=np.concatenate([np.full(len(e['y']),1/len(domains)/counts[e['domain']]/len(e['y'])) for e in episodes])
        y=np.concatenate([e['y']-e['predictions'][group] for e in episodes]);self.mean=float(w@y)
        self.source_keys=[(e['cell_id'],int(j)) for e in episodes for j in e['query_indices']]
        if self.learner=='constant':return self
        x=np.concatenate([features(byid[e['cell_id']],e['predictions'][group],e['query_indices'],self.view) for e in episodes])
        self.keep=np.flatnonzero(x.std(0)>1e-8);assert len(self.keep)
        self.scaler=StandardScaler().fit(x[:,self.keep],sample_weight=w);z=self.scaler.transform(x[:,self.keep])
        self.model=Ridge(alpha=self.alpha) if self.learner=='ridge' else KernelRidge(alpha=self.alpha,kernel='rbf',gamma=1/z.shape[1])
        self.model.fit(z,y-self.mean,sample_weight=w*len(w));return self

    def predict(self,cell,pred,indices=None):
        if indices is None:indices=np.arange(sa.K,len(cell.x))
        r=np.full(len(pred),self.mean)
        if self.learner!='constant':
            x=features(sa.inference_view(cell),pred,indices,self.view)
            r+=self.model.predict(self.scaler.transform(x[:,self.keep]))
        return np.clip(r,-self.clip,self.clip)
