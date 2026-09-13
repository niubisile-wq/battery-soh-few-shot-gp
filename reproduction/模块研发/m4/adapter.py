"""Deployable bounded residual adapter, with explicit target-label masking."""
import numpy as np
from source_oof import sa


class ResidualM4Adapter:
    def __init__(self,parent,corrector,gain,parent_mode=None):
        if not np.isfinite(gain) or not 0<=gain<=1:
            raise ValueError('M4 gain must be finite and in [0,1]')
        self.parent,self.corrector,self.gain,self.parent_mode=parent,corrector,float(gain),parent_mode

    def predict(self,cell):
        c=sa.inference_view(cell)
        p=self.parent.predict(c) if self.parent_mode is None else self.parent.predict(c,self.parent_mode)
        p=np.asarray(p,dtype=float)
        if p.shape!=(len(c.x)-sa.K,) or not np.isfinite(p).all():
            raise ValueError('Invalid parent prediction')
        if self.gain==0:return p.copy()
        r=np.asarray(self.corrector.predict(c,p),dtype=float)
        if r.shape!=p.shape or not np.isfinite(r).all():
            raise ValueError('Invalid residual prediction')
        return p+self.gain*r
