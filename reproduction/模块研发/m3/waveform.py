"""Full-window reference-paired waveform encoding for a source-budgeted GP."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF,Matern,WhiteKernel
from geometry import K


def waveform_features(cell,view):
    x=np.asarray(cell.x,dtype=float).copy()
    assert len(x)>=K and np.isfinite(x).all() and cell.reference_capacity>0
    x[:,0]-=3.5
    x[:,1]/=cell.reference_capacity
    x[:,2]=(x[:,2]-x[:,2,:1])/3600
    absolute=x.reshape(len(x),-1)
    if view=='wave_absolute':return absolute
    if view=='wave_pair':return np.c_[absolute,absolute-absolute[:K].mean(axis=0)]
    raise ValueError(view)


class WaveformGP:
    def __init__(self,view,kernel):
        self.view,self.kernel=view,kernel
        self.raw_view,self.encoder=view.rsplit('_',1)

    def feature_matrix(self,cell):
        raw=waveform_features(cell,self.raw_view)[:,self.keep]
        return self.projection.transform(self.raw_scaler.transform(raw))[:,self.latent_keep]

    def fit(self,cells,allowed):
        xx=[];yy=[];self.source_keys=[]
        for c in cells:
            idx=np.asarray(allowed[c.id],dtype=int);assert set(range(K))<=set(idx)
            xx.append(waveform_features(c,self.raw_view)[idx]);yy.append(c.y[idx]-np.mean(c.y[:K]))
            self.source_keys.extend((c.id,int(j)) for j in idx)
        x,y=np.concatenate(xx),np.concatenate(yy);assert len(y)<=1000 and np.isfinite(y).all()
        floor=np.repeat([1e-6,1e-4,1e-7],128)
        if self.raw_view=='wave_pair':floor=np.tile(floor,2)
        self.keep=np.flatnonzero(x.std(axis=0)>floor);assert len(self.keep)>0
        self.raw_scaler=StandardScaler().fit(x[:,self.keep]);z=self.raw_scaler.transform(x[:,self.keep])
        if self.encoder=='pca8':
            self.projection=PCA(n_components=min(8,z.shape[1],len(z)-1),svd_solver='full').fit(z)
        elif self.encoder=='pls4':
            self.projection=PLSRegression(n_components=min(4,z.shape[1],len(z)-1),scale=False,max_iter=1000).fit(z,y)
        else:raise ValueError(self.encoder)
        latent=self.projection.transform(z)
        # Do not amplify numerical null-space components into false information.
        self.latent_keep=np.flatnonzero(latent.std(axis=0)>1e-6);assert len(self.latent_keep)>0
        self.scaler=StandardScaler().fit(latent[:,self.latent_keep])
        base=RBF(1.,(1e-2,1e3)) if self.kernel=='rbf' else Matern(1.,(1e-2,1e3),nu=1.5)
        self.gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(1e-3,1e3))*base+WhiteKernel(.01,(1e-6,1)),normalize_y=True,random_state=0)
        self.gp.fit(self.scaler.transform(latent[:,self.latent_keep]),y)
        return self
