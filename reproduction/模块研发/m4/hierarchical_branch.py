"""Budgeted two-level source covariance with fixed uniform target adaptation."""
from copy import deepcopy
import numpy as np
from difference_branch import DifferenceBranch
from mixed_effects_gp import MixedEffectsGP
from hierarchical_slopes import HierarchicalSlopesGP
from evidence_mixed import EvidenceMixed
from source_oof import sa


def fit_branch(reference,cells,allowed,hierarchical=True,optimize=True):
    result=DifferenceBranch('difference',reference.kernel,optimize)
    for attr in ('keep','raw_scaler','pca','latent_keep','scaler','source_keys','scale'):
        setattr(result,attr,deepcopy(getattr(reference,attr)))
    byid={c.id:c for c in cells}
    assert set(byid)==set(allowed)
    assert set(result.source_keys)=={(cid,int(i)) for cid,indices in allowed.items() for i in indices}
    assert len(result.source_keys)==sum(map(len,allowed.values()))<=1000
    assert all(set(range(sa.K))<=set(ix) for ix in allowed.values())
    y=np.array([byid[cid].y[j]-np.mean(byid[cid].y[:sa.K]) for cid,j in result.source_keys])
    assert np.isfinite(y).all()
    ids=[cid for cid,j in result.source_keys];domains=[byid[cid].domain for cid in ids]
    kernel=deepcopy(reference.gp.kernel)
    if hierarchical:
        result.gp=HierarchicalSlopesGP(kernel,optimize).fit(reference.gp.X_train_,y,ids,domains)
    else:result.gp=MixedEffectsGP(kernel,optimize).fit(reference.gp.X_train_,y,ids)
    return EvidenceMixed(result,'uniform')
