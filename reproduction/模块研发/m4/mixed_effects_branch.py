"""Reuse an audited label-free source encoder; refit only source covariance model."""
from copy import deepcopy
import numpy as np
from difference_branch import DifferenceBranch
from mixed_effects_gp import MixedEffectsGP
from source_oof import sa


class MixedEffectsBranch(DifferenceBranch):
    @classmethod
    def from_encoder(cls,reference,cells,allowed,learn_rho=True,optimize=True):
        assert reference.target=='difference'
        result=cls(target='difference',kernel=reference.kernel,optimize=optimize)
        for attr in ('keep','raw_scaler','pca','latent_keep','scaler','source_keys','scale'):
            setattr(result,attr,deepcopy(getattr(reference,attr)))
        byid={c.id:c for c in cells}
        assert set(byid)==set(allowed)
        assert set(result.source_keys)=={(cid,int(j)) for cid,ix in allowed.items() for j in ix}
        assert len(result.source_keys)==sum(map(len,allowed.values()))<=1000
        assert all(set(range(sa.K))<=set(ix) for ix in allowed.values())
        y=np.array([byid[cid].y[j]-np.mean(byid[cid].y[:sa.K]) for cid,j in result.source_keys])
        assert np.isfinite(y).all()
        # Kernel starts from original prior parameters, not fitted difference optimum.
        result.gp=MixedEffectsGP(kernel=deepcopy(reference.gp.kernel),optimize=optimize,
            rho=.1 if learn_rho else 0.,learn_rho=learn_rho).fit(
                reference.gp.X_train_,y,[cid for cid,j in result.source_keys])
        return result
