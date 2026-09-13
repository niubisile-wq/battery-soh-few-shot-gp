"""Reference-voltage registered cumulative partial charge, no future landmarks."""
import numpy as np
from multiscale import MultiscaleGP
from source_oof import sa


def charge_features(cell):
    x=np.asarray(cell.x,float);assert len(x)>=sa.K and x.shape[1:]==(3,128) and np.isfinite(x).all()
    v,current,time=x[:,0],x[:,1],x[:,2];dt=np.diff(time,axis=1)
    assert dt.min()>=-1e-6 and cell.reference_capacity>0
    increments=np.maximum(.5*(current[:,:-1]+current[:,1:])*np.maximum(dt,0)/3600,0)
    charge=np.c_[np.zeros(len(x)),np.cumsum(increments,axis=1)]
    assert np.all(charge[:,-1]>0)
    lo=float(np.mean(np.min(v[:sa.K],axis=1)));hi=float(np.mean(np.max(v[:sa.K],axis=1)))
    assert hi>lo+1e-8
    grid=np.linspace(lo,hi,64);profiles=[];coverage=[]
    for voltage,q in zip(v,charge):
        envelope=np.maximum.accumulate(voltage)
        # Keep final charge at a voltage plateau; policy fixed before evaluation.
        index=np.r_[np.flatnonzero(np.diff(envelope)>0),len(envelope)-1]
        u=envelope[index];assert len(u)>=2
        curve=np.interp(grid,u,q[index])
        profiles.append((curve-curve[0])/cell.reference_capacity)
        coverage.append(((grid>=u[0])&(grid<=u[-1])).astype(float))
    absolute=np.asarray(profiles);relative=absolute-absolute[:sa.K].mean(0)
    duration=np.maximum(time[:,-1]-time[:,0],0)/3600
    context=np.c_[np.full(len(x),lo),np.full(len(x),hi),v[:,0],v[:,-1],
                  current.mean(1)/cell.reference_capacity,duration]
    result=np.c_[absolute,relative,coverage,context];assert np.isfinite(result).all()
    return result


class RegisteredChargeGP(MultiscaleGP):
    def raw_features(self,cell):
        return charge_features(cell)
