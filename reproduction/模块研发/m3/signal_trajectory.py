"""Causal signal-conditioned trend estimate using only rolling observed moments."""
import numpy as np
from trajectory import rolling_sum,K


def correction(cell,p,kind,clip=.05):
    p=np.asarray(p,dtype=float)
    assert p.shape==(len(cell.x)-K,)
    if kind=='identity':return np.zeros_like(p)
    _,width,shrink=kind.split('_');width=int(width);shrink=float(shrink)
    y=np.r_[cell.y[:K],p].astype(float);t=cell.cycle.astype(float)-float(cell.cycle[0])
    x=cell.x.astype(float);dt=np.diff(x[:,2],axis=1)
    assert np.min(dt)>=-1e-6
    q=np.sum(.5*(x[:,1,1:]+x[:,1,:-1])*np.maximum(dt,0),axis=1)/3600
    assert np.all(q>0) and np.isfinite(y).all()
    q=np.log(q/np.mean(q[:K]))
    n=np.minimum(np.arange(1,len(y)+1),width)
    mean=lambda a:rolling_sum(a,width)/n
    mt,my,mq=mean(t),mean(y),mean(q)
    vt=np.maximum(mean(t*t)-mt*mt,0);vq=np.maximum(mean(q*q)-mq*mq,0)
    cty=mean(t*y)-mt*my;ctq=mean(t*q)-mt*mq;cqy=mean(q*y)-mq*my
    divide=lambda a,b:np.divide(a,b,out=np.zeros_like(a),where=b>1e-12)
    sy=divide(cty,vt);sq=divide(ctq,vt)
    residual_variance=np.maximum(vq-sq*ctq,0)
    coefficient=divide(cqy-sq*cty,residual_variance+shrink*vq+1e-12)
    estimate=my+sy*(t-mt)+coefficient*(q-mq-sq*(t-mt))
    return np.clip(estimate[K:]-p,-clip,clip)
