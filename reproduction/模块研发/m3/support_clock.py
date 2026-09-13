"""Reference-calibrated degradation coordinates using only first-K SOH labels."""
import numpy as np
from geometry import features,K
from progression import ProgressionGP


def reference_trend(cell):
    t=np.asarray(cell.cycle[:K],dtype=float);y=np.asarray(cell.y[:K],dtype=float)
    assert len(t)==K and np.isfinite(y).all() and np.all(np.diff(t)>0)
    i,j=np.triu_indices(K,1)
    slope=float(np.median((y[j]-y[i])/(t[j]-t[i])))
    anchor=float(np.mean(y));center=float(np.mean(t))
    residual=y-anchor-slope*(t-center)
    noise=max(1.4826*float(np.median(abs(residual-np.median(residual)))),1e-5)
    uncertainty=noise/max(float(np.sqrt(np.sum((t-center)**2))),1.)
    return anchor,center,slope,noise,uncertainty


def clock_features(cell,view):
    anchor,center,slope,noise,uncertainty=reference_trend(cell)
    t=np.asarray(cell.cycle,dtype=float);elapsed=t-t[0];span=max(float(t[K-1]-t[0]),1.)
    clock=np.c_[np.log1p(elapsed),np.log1p(elapsed/span),np.arcsinh(elapsed*slope/.05),
                np.arcsinh(elapsed*uncertainty/.05),np.full(len(t),slope*span),np.full(len(t),noise)]
    if view=='clock_only':return clock
    if view=='clock_joint':return np.c_[clock,features(cell,'relative_absolute')]
    raise ValueError(view)


class SupportClockGP(ProgressionGP):
    def feature_matrix(self,cell):return clock_features(cell,self.view)


def ordinary_trend(cell):
    anchor,center,slope,_,_=reference_trend(cell)
    return anchor+slope*(np.asarray(cell.cycle[K:],dtype=float)-center)
