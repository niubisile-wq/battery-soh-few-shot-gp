"""Residual regression learned from budgeted source-domain-held-out episodes."""
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

REGULARIZATION=[.1,1.,10.,100.,1000.]


def features(e,kind):
    b=np.asarray(e['base']);r=np.full(len(b),e['residual'])
    s=np.full(len(b),np.log(e['parent_support_mse']+1e-6))
    v=np.log(np.asarray(e['parent_variance'])+1e-6)
    x=np.c_[b,r,b*r,s,v]
    if kind=='dual':
        p=np.asarray(e['physical']);d=p-b
        x=np.c_[x,p,d,d*b,np.log(e['physical_variance']+1e-6),
                np.full(len(b),np.log(e['physical_support_mse']+1e-6))]
    elif kind!='parent': raise ValueError(kind)
    assert np.isfinite(x).all()
    return x


def fit(episodes):
    counts=Counter(e['domain'] for e in episodes)
    weights=np.concatenate([np.full(len(e['base']),1/(counts[e['domain']]*len(e['base']))) for e in episodes])
    weights*=len(weights)/weights.sum()
    y=np.concatenate([e['y']-e['base'] for e in episodes])
    heads={}
    for kind in ['parent','dual']:
        x=np.concatenate([features(e,kind) for e in episodes])
        scaler=StandardScaler().fit(x,sample_weight=weights)
        z=scaler.transform(x)
        models=[Ridge(alpha=a).fit(z,y,sample_weight=weights) for a in REGULARIZATION]
        heads[kind]=(scaler,models)
    return heads


def attach(episode,heads):
    episode['meta_predictions']={kind:np.stack([m.predict(scaler.transform(features(episode,kind))) for m in models])
                                 for kind,(scaler,models) in heads.items()}
    return episode
