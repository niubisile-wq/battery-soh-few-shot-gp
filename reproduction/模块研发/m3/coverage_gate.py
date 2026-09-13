"""Source-only normalized kernel coverage, never conditioned on query truth."""
import numpy as np
from geometry import K


def coverage(model,cell):
    parent=model.parent;gp=parent.gp
    z=parent.scaler.transform(parent.feature_matrix(cell))
    signal=gp.kernel_.k1
    train=gp.X_train_;support=np.array([j<K for cid,j in model.source_keys])
    assert support.any() and len(support)==len(train)
    scale=np.sqrt(np.maximum(signal.diag(train),1e-30))
    query=[]
    for start in range(K,len(z),256):
        q=z[start:start+256]
        corr=signal(q,train)/np.sqrt(np.maximum(signal.diag(q),1e-30))[:,None]/scale[None,:]
        query.append(np.max(corr,axis=1))
    ref=signal(z[:K],train[support])/np.sqrt(np.maximum(signal.diag(z[:K]),1e-30))[:,None]/scale[support][None,:]
    return np.clip(np.concatenate(query),0,1),float(np.clip(np.mean(np.max(ref,axis=1)),0,1))


def trust(query,reference,kind,power):
    if kind=='none':return np.ones_like(query)
    if kind=='query':return query**power
    if kind=='product':return (query*reference)**power
    raise ValueError(kind)
