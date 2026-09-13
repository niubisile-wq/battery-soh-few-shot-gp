"""First-K conditional adaptation of a frozen source relative-geometry GP."""
import numpy as np
from scipy.linalg import cho_solve
from geometry import features,K


class ConditionalGeometry:
    def __init__(self,parent,mode):
        self.parent,self.mode=parent,mode
        self.source_keys=parent.source_keys

    def predict(self,cell):
        model=self.parent;gp=model.gp
        z=model.scaler.transform(model.feature_matrix(cell) if hasattr(model,'feature_matrix') else features(cell,model.view))
        p=np.concatenate([gp.predict(z[j:j+256]) for j in range(0,len(z),256)])
        anchor=float(np.mean(cell.y[:K]))
        if self.mode=='source_only':return anchor+p[K:]
        if self.mode=='source_bias':return anchor+p[K:]-float(np.mean(p[:K]))
        if self.mode!='posterior':raise ValueError(self.mode)
        zs=z[:K];cross=gp.kernel_(gp.X_train_,zs)
        solved=cho_solve((gp.L_,True),cross,check_finite=False)
        ss=gp.kernel_(zs)-cross.T@solved
        ss=.5*(ss+ss.T)+np.eye(K)*1e-9
        residual=cell.y[:K]-anchor-p[:K]
        weights=np.linalg.solve(ss,residual)
        output=[]
        for j in range(K,len(z),256):
            zz=z[j:j+256]
            qs=gp.kernel_(zz,zs)-gp.kernel_(zz,gp.X_train_)@solved
            output.append(anchor+p[j:j+256]+qs@weights)
        result=np.concatenate(output)
        assert np.isfinite(result).all()
        return result
