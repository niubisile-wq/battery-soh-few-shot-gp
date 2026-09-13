"""Local-charge residual correction with a matched condition-removal option."""
import numpy as np
from collections import OrderedDict
import hashlib
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from source_oof import sa
from features import hi
from geometry import features as geometry_features
from correction import features as state_features


_RAW_CACHE=OrderedDict()


def parts(cell,indices):
    # Raw deterministic features only: no fitted transforms, predictions or labels in cache.
    indices=np.asarray(indices,dtype=int)
    x=np.ascontiguousarray(cell.x[np.r_[np.arange(sa.K),indices]])
    key=(x.shape,x.dtype.str,hashlib.sha256(x.tobytes()).digest())
    if key in _RAW_CACHE:
        _RAW_CACHE.move_to_end(key);return _RAW_CACHE[key]
    h=hi(x);ref=h[:sa.K].mean(0);v=np.asarray(x[:,0],float)
    context=np.c_[np.broadcast_to(ref,(len(indices),len(ref))),h[sa.K:,7:14],v[sa.K:,0],v[sa.K:,-1]-v[sa.K:,0]]
    result=geometry_features(SimpleNamespace(x=x),'relative')[sa.K:],context
    _RAW_CACHE[key]=result
    if len(_RAW_CACHE)>64:_RAW_CACHE.popitem(last=False)
    return result


class ConditionCorrector:
    def __init__(self,view='state',learner='constant',alpha=1.,clip=.05):
        self.view,self.learner,self.alpha,self.clip=view,learner,alpha,clip

    def fit(self,episodes,byid,group):
        self.source_keys=[(e['cell_id'],int(j)) for e in episodes for j in e['query_indices']]
        domains={e['domain'] for e in episodes};counts={d:sum(e['domain']==d for e in episodes) for d in domains}
        w=np.concatenate([np.full(len(e['y']),1/len(domains)/counts[e['domain']]/len(e['y'])) for e in episodes])
        y=np.concatenate([e['y']-e['predictions'][group] for e in episodes]);self.mean=float(w@y)
        if self.learner=='constant':return self
        gg=[];cc=[];ss=[]
        for e in episodes:
            c=byid[e['cell_id']];g,z=parts(c,e['query_indices']);gg.append(g);cc.append(z)
            ss.append(state_features(c,e['predictions'][group],e['query_indices'],'state'))
        g=np.concatenate(gg);context=np.concatenate(cc);state=np.concatenate(ss)
        if self.view=='geometry_residualized':
            self.context_keep=np.flatnonzero(context.std(0)>1e-8);assert len(self.context_keep)
            self.context_scaler=StandardScaler().fit(context[:,self.context_keep],sample_weight=w)
            self.condition_model=Ridge(alpha=10).fit(self.context_scaler.transform(context[:,self.context_keep]),g,sample_weight=w*len(w))
            g=g-self.condition_model.predict(self.context_scaler.transform(context[:,self.context_keep]))
        elif self.view!='geometry_raw':raise ValueError(self.view)
        x=np.c_[state,g];self.keep=np.flatnonzero(x.std(0)>1e-8);assert len(self.keep)
        self.scaler=StandardScaler().fit(x[:,self.keep],sample_weight=w);z=self.scaler.transform(x[:,self.keep])
        self.model=Ridge(alpha=self.alpha) if self.learner=='ridge' else KernelRidge(alpha=self.alpha,kernel='rbf',gamma=1/z.shape[1])
        self.model.fit(z,y-self.mean,sample_weight=w*len(w));return self

    def predict(self,cell,pred,indices=None):
        if self.learner=='constant':return np.full(len(pred),np.clip(self.mean,-self.clip,self.clip))
        if indices is None:indices=np.arange(sa.K,len(cell.x))
        c=sa.inference_view(cell);g,context=parts(c,indices)
        if self.view=='geometry_residualized':g=g-self.condition_model.predict(self.context_scaler.transform(context[:,self.context_keep]))
        x=np.c_[state_features(c,pred,indices,'state'),g]
        r=self.mean+self.model.predict(self.scaler.transform(x[:,self.keep]))
        return np.clip(r,-self.clip,self.clip)
