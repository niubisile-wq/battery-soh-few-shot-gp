"""Support-initialized causal level/trend filter; no future queries or labels."""
import numpy as np


def filter_prediction(pred,cycles,anchor,anchor_cycle,tau,gain,clip):
    if tau==0: return np.asarray(pred).copy()
    level=float(anchor);trend=0.;last=float(anchor_cycle);out=[]
    for p,t in zip(pred,cycles,strict=True):
        dt=float(t-last)
        if dt<=0: raise ValueError('Non-increasing query cycles')
        a=-np.expm1(-dt/tau)
        prior=level+trend*dt
        residual=float(p)-prior
        if clip: residual=float(np.clip(residual,-clip,clip))
        level=prior+a*residual
        trend=(1-gain*a)*trend+gain*a*(level-(prior-trend*dt))/dt
        out.append(level);last=float(t)
    return np.asarray(out)
