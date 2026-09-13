"""Transfer only the first-ten posterior update, not the source branch level."""
import numpy as np
from support_cv_mixed import SupportCVMixed


class InnovationAdapter:
    def __init__(self,parent,mode='cv'):
        self.parent=parent;self.adapter=SupportCVMixed(parent,mode)
        self.source_keys=parent.source_keys

    def predict(self,c):
        # Both predictions have the identical first-ten mean anchoring.
        return self.adapter.predict(c)-self.parent.predict(c)

    def apply(self,c,parent_prediction,gain):
        assert 0<=gain<=1
        p=np.asarray(parent_prediction,float);delta=self.predict(c)
        assert p.shape==delta.shape and np.isfinite(p).all()
        return p+gain*delta
