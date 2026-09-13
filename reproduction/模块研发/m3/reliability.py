"""Bounded fusion weight learned from source OOF error, not query truth."""
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from geometry import features,K
from support_clock import clock_features


class ReliabilityGate:
    def __init__(self,view,alpha):self.view,self.alpha=view,alpha

    def inputs(self,c,p,b,idx):
        anchor=float(np.mean(c.y[:K]));delta=b-p
        state=np.c_[p-anchor,b-anchor,delta,abs(delta),clock_features(c,'clock_only')[idx]]
        return state if self.view=='state' else np.c_[state,features(c,'relative_absolute')[idx]]

    def fit(self,episodes,branches,cells,allowed):
        byid={c.id:c for c in cells};counts=Counter(e['domain'] for e in episodes)
        xx=[];tt=[];ww=[];self.source_keys=[]
        for e in episodes:
            c=byid[e['cell_id']];idx=np.array([j for j in allowed[c.id] if j>=K]);b=branches[c.id];p=e['base']
            assert np.array_equal(c.y[idx],e['y']) and len(b)==len(idx)
            d=b-p;v=d*d+.01**2
            target=np.clip(d*(e['y']-p)/v,0,.75)
            xx.append(self.inputs(c,p,b,idx));tt.append(target);ww.append(v/(counts[e['domain']]*len(idx)))
            self.source_keys.extend((c.id,int(j)) for j in idx)
        x,t,w=np.concatenate(xx),np.concatenate(tt),np.concatenate(ww);w=w/w.mean()
        assert np.isfinite(x).all() and np.isfinite(t).all()
        self.scaler=StandardScaler().fit(x,sample_weight=w)
        self.regressor=Ridge(alpha=self.alpha).fit(self.scaler.transform(x),t,sample_weight=w)
        self.constant=float(np.average(t,weights=w));return self

    def predict(self,c,p,b):
        idx=np.arange(K,len(c.x));assert len(p)==len(b)==len(idx)
        return np.clip(self.regressor.predict(self.scaler.transform(self.inputs(c,p,b,idx))),0,.75)
