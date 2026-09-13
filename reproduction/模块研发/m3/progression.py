"""Causal cycle-progress coordinates with matched progress-only GP control."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF,Matern,WhiteKernel
from geometry import features,K


def progression_features(cell,view):
    cycle=np.asarray(cell.cycle,dtype=float)
    assert len(cycle)>=K and np.all(np.diff(cycle)>0)
    elapsed=cycle-cycle[0]
    support_span=max(float(cycle[K-1]-cycle[0]),1.)
    age=np.c_[np.log1p(elapsed),np.log1p(elapsed/support_span)]
    if view=='progress_only':return age
    if view=='progress_relative':return np.c_[age,features(cell,'relative')]
    if view=='progress_absolute':return np.c_[age,features(cell,'relative_absolute')]
    raise ValueError(view)


class ProgressionGP:
    def __init__(self,view,kernel):self.view,self.kernel=view,kernel

    def feature_matrix(self,cell):return progression_features(cell,self.view)

    def fit(self,cells,allowed):
        xx=[];yy=[];self.source_keys=[]
        for c in cells:
            idx=np.asarray(allowed[c.id],dtype=int);assert set(range(K))<=set(idx)
            xx.append(self.feature_matrix(c)[idx]);yy.append(c.y[idx]-np.mean(c.y[:K]))
            self.source_keys.extend((c.id,int(j)) for j in idx)
        x,y=np.concatenate(xx),np.concatenate(yy)
        assert np.isfinite(x).all() and np.isfinite(y).all() and len(y)<=1000
        self.scaler=StandardScaler().fit(x)
        base=RBF(1.,(1e-2,1e3)) if self.kernel=='rbf' else Matern(1.,(1e-2,1e3),nu=1.5)
        self.gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(1e-3,1e3))*base+WhiteKernel(.01,(1e-6,1)),
            normalize_y=True,random_state=0)
        self.gp.fit(self.scaler.transform(x),y);return self

    def predict(self,cell):
        z=self.scaler.transform(self.feature_matrix(cell))
        p=np.concatenate([self.gp.predict(z[j:j+256]) for j in range(0,len(z),256)])
        return float(np.mean(cell.y[:K]))+p[K:]-float(np.mean(p[:K]))
