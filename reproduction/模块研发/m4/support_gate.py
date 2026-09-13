"""Source-only neighborhood support attenuation; not calibrated confidence."""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from source_oof import sa


class SupportGate:
    def __init__(self, model):
        self.model=model
        z=np.asarray(model.gp.X_train_,float)
        ids=np.array([cid for cid,_ in model.source_keys])
        assert len(ids)==len(z) and len(set(ids))>1
        distances=euclidean_distances(z,z,squared=True)
        distances[ids[:,None]==ids[None,:]]=np.inf
        nearest=distances.min(axis=1)
        # Equal-cell median avoids using same-cell repetitions as support evidence.
        self.cell_scales={cid:float(np.median(nearest[ids==cid])) for cid in sorted(set(ids))}
        self.scale2=max(float(np.median(list(self.cell_scales.values()))),1e-8)

    def predict(self,cell,multiplier=1.):
        if multiplier<=0:raise ValueError(multiplier)
        visible=sa.inference_view(cell)
        z=self.model.scaler.transform(self.model.feature_matrix(visible))[sa.K:]
        nearest=np.concatenate([euclidean_distances(z[j:j+256],self.model.gp.X_train_,squared=True).min(axis=1) for j in range(0,len(z),256)])
        return 1/(1+nearest/(self.scale2*multiplier))
