"""Strictly causal filters of frozen predictions, initialized by allowed support."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

K=10


def rolling_sum(x,width):
    s=np.r_[0.,np.cumsum(np.asarray(x,dtype=float))]
    end=np.arange(1,len(x)+1);return s[end]-s[np.maximum(end-width,0)]


def correction(cell,p,kind,clip=.05):
    p=np.asarray(p,dtype=float)
    assert p.shape==(len(cell.y)-K,) and np.isfinite(p).all()
    if kind=='identity':return np.zeros_like(p)
    y=np.r_[cell.y[:K],p].astype(float);t=cell.cycle.astype(float)-float(cell.cycle[0])
    assert np.isfinite(y).all() and np.all(np.diff(t)>0)
    if kind.startswith('linear_'):
        width=int(kind.split('_')[1]);n=np.minimum(np.arange(1,len(y)+1),width)
        st=rolling_sum(t,width);sy=rolling_sum(y,width)
        denom=rolling_sum(t*t,width)-st*st/n
        slope=np.divide(rolling_sum(t*y,width)-st*sy/n,denom,out=np.zeros_like(y),where=denom>1e-10)
        result=sy/n+slope*(t-st/n)
    elif kind.startswith('median_'):
        width=int(kind.split('_')[1])
        result=np.median(sliding_window_view(np.pad(y,(width-1,0),mode='edge'),width),axis=1)
    elif kind.startswith('ema_'):
        alpha=float(kind.split('_')[1]);result=np.empty_like(y);result[0]=y[0]
        for j in range(1,len(y)):result[j]=alpha*y[j]+(1-alpha)*result[j-1]
    elif kind.startswith('holt_'):
        _,alpha,beta=kind.split('_');alpha,beta=float(alpha),float(beta)
        result=np.empty_like(y);result[0]=y[0];trend=0.
        for j in range(1,len(y)):
            dt=t[j]-t[j-1]
            result[j]=alpha*y[j]+(1-alpha)*(result[j-1]+trend*dt)
            trend=beta*(result[j]-result[j-1])/dt+(1-beta)*trend
    else:raise ValueError(kind)
    return np.clip(result[K:]-p,-clip,clip)
