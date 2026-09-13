"""Conditional source-LODO error regressor, no query labels at inference."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from geometry import features, K


class ResidualModel:
    def __init__(self, view, learner):
        self.view,self.learner=view,learner

    def inputs(self,c,p,indices):
        return np.c_[features(c,self.view)[indices],np.asarray(p)-float(np.mean(c.y[:K]))]

    def fit(self,episodes,cells,allowed):
        byid={c.id:c for c in cells};xx=[];yy=[];self.source_keys=[]
        for e in episodes:
            c=byid[e['cell_id']];idx=np.asarray([j for j in allowed[c.id] if j>=K])
            assert np.array_equal(e['y'],c.y[idx])
            xx.append(self.inputs(c,e['base'],idx));yy.append(e['y']-e['base'])
            self.source_keys.extend((c.id,int(j)) for j in idx)
        x,y=np.concatenate(xx),np.concatenate(yy)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        self.scaler=StandardScaler().fit(x)
        self.model=Ridge(alpha=10) if self.learner=='ridge' else GaussianProcessRegressor(
            kernel=ConstantKernel(1.,(1e-3,1e3))*RBF(1.,(1e-2,1e3))+WhiteKernel(.01,(1e-6,1)),
            normalize_y=True,random_state=0)
        self.model.fit(self.scaler.transform(x),y)
        return self

    def predict(self,c,p):
        assert len(p)==len(c.x)-K
        x=self.scaler.transform(self.inputs(c,p,np.arange(K,len(c.x))))
        return np.concatenate([self.model.predict(x[j:j+256]) for j in range(0,len(x),256)])
