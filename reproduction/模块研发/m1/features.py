"""Causal cell-reference views and source-only learned feature transformations."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from data import K


def hi(x):
    # Preserve float32 arithmetic of the original champion exactly.
    grid=np.linspace(0,1,x.shape[-1],dtype='float32');grid-=grid.mean()
    out=[]
    for ch in range(x.shape[1]):
        z=x[:,ch,:]
        slope=((z-z.mean(1,keepdims=True))*grid).sum(1)/(grid**2).sum()
        out.append(np.stack([z.mean(1),z.std(1),z.min(1),z.max(1),z[:,0],z[:,-1],slope],1))
    return np.concatenate(out,1).astype('float64')


def physics(cell):
    x=cell.x.astype('float64');v,current,t=x[:,0],x[:,1],x[:,2]
    increments=.5*(current[:,1:]+current[:,:-1])*np.diff(t,axis=1)/3600
    q=np.c_[np.zeros(len(x)),np.cumsum(increments,axis=1)]
    if (increments<0).any() or (q[:,-1]<=0).any():
        raise ValueError('Physical view requires observed positive charging')
    fraction=q/q[:,-1,None]
    grid=np.linspace(.125,.875,7)
    vv=np.stack([np.interp(grid,fraction[j],v[j]) for j in range(len(x))])
    span=v[:,-1]-v[:,0]
    if (span<=0).any():raise ValueError('Nonpositive window voltage span')
    shape=(vv-v[:,0,None])/span[:,None]
    return np.c_[q[:,-1]/cell.reference_capacity,
                 current.mean(1)/cell.reference_capacity,
                 current.std(1)/np.maximum(current.mean(1),1e-8),
                 v[:,0],span,shape]


def raw_view(cell,view):
    base=physics(cell) if view.startswith('physics') else hi(cell.x)
    if view.endswith('_dual'):
        return np.c_[base,base-base[:K].mean(0,keepdims=True)]
    return base


def target_shift(cell,anchored):
    return float(cell.y[:K].mean()) if anchored else 0.


class FeatureMap:
    def __init__(self,spec,config):
        self.spec=spec;self.config=config;self.pls=None

    def fit(self,cells,indices):
        raw=np.concatenate([raw_view(c,self.spec['view'])[indices[c.id]] for c in cells])
        self.keep=np.arange(raw.shape[1])
        if self.spec.get('prune'):
            # Absolute physical resolution thresholds, fixed before outer scoring.
            if self.spec['view'].startswith('hi'):
                floor=np.r_[np.full(7,1e-6),np.full(7,1e-4),np.full(7,1e-3)]
                if self.spec['view'].endswith('_dual'):floor=np.r_[floor,floor]
            else:
                floor=np.full(raw.shape[1],1e-7)
            keep=np.flatnonzero(raw.std(0)>floor)
            selected=[]
            for i in keep:
                if not selected or np.max(np.abs(np.corrcoef(raw[:,selected+[i]].T)[-1,:-1]))<.999:
                    selected.append(int(i))
            self.keep=np.asarray(selected)
            if not len(self.keep):raise ValueError('Pruning removed all source features')
        self.scaler=StandardScaler().fit(raw[:,self.keep])
        self.weights=np.ones(len(self.keep))
        if self.spec['view'].endswith('_dual'):
            self.weights[self.keep>=raw.shape[1]//2]=self.config.get('delta_weight',1.)
        z=self.scaler.transform(raw[:,self.keep])*self.weights
        if self.spec.get('metric')=='pls':
            y=np.concatenate([c.y[indices[c.id]]-target_shift(c,self.spec.get('anchored',False)) for c in cells])
            self.pls=PLSRegression(n_components=min(self.config['components'],z.shape[1]),scale=False,max_iter=1000)
            self.pls.fit(z,y)
            self.latent_scaler=StandardScaler().fit(self.pls.transform(z))
        elif self.spec.get('metric') in ('pls_concat','pls_poly'):
            y=np.concatenate([c.y[indices[c.id]]-target_shift(c,self.spec.get('anchored',False)) for c in cells])
            self.pls=PLSRegression(n_components=min(self.config['components'],z.shape[1]),scale=False,max_iter=1000)
            self.pls.fit(z,y)
            self.latent_scaler=StandardScaler().fit(self.pls.transform(z))
            if self.spec.get('metric')=='pls_poly':
                latent=self.latent_scaler.transform(self.pls.transform(z))
                self.poly_scaler=StandardScaler().fit(latent**2)
        elif self.spec.get('metric')=='mlp_concat':
            y=np.concatenate([c.y[indices[c.id]]-target_shift(c,self.spec.get('anchored',False)) for c in cells])
            self.mlp=MLPRegressor(hidden_layer_sizes=(self.config['hidden'],), activation='tanh',
                                  solver='lbfgs', alpha=self.config['alpha'], max_iter=500,
                                  random_state=0)
            self.mlp.fit(z,y)
            hidden=np.tanh(z@self.mlp.coefs_[0]+self.mlp.intercepts_[0])
            self.hidden_scaler=StandardScaler().fit(hidden)
        return self

    def transform(self,cell):
        z=self.scaler.transform(raw_view(cell,self.spec['view'])[:,self.keep])*self.weights
        if self.pls is not None:
            latent=self.latent_scaler.transform(self.pls.transform(z))
            if self.spec.get('metric')=='pls_concat':
                z=np.c_[z,latent]
            elif self.spec.get('metric')=='pls_poly':
                poly=self.poly_scaler.transform(latent**2)
                z=np.c_[z,latent,poly*self.config.get('poly_weight',1.0)]
            else:
                z=latent
        elif getattr(self,'mlp',None) is not None:
            hidden=np.tanh(z@self.mlp.coefs_[0]+self.mlp.intercepts_[0])
            z=np.c_[z,self.hidden_scaler.transform(hidden)]
        return z

    def serialize(self):
        return {'source_columns':self.keep.tolist(),'source_mean':self.scaler.mean_.tolist(),
                'source_scale':self.scaler.scale_.tolist(),'block_weights':self.weights.tolist(),
                'pls_components':self.pls.n_components if self.pls is not None else None}
