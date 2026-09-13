"""Local signed Haar shape and detail energies for a budgeted direct SOH GP."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from source_oof import sa


def local_features(cell, paired=True):
    x=np.asarray(cell.x,dtype=float).copy()
    assert x.ndim==3 and x.shape[1:]==(3,128) and len(x)>=sa.K
    assert np.isfinite(x).all() and cell.reference_capacity>0
    x[:,0]-=3.5; x[:,1]/=cell.reference_capacity
    x[:,2]=(x[:,2]-x[:,2,:1])/3600
    approximation=x; details=[]
    for _ in range(5):
        a,b=approximation[...,::2],approximation[...,1::2]
        details.append((a-b)/np.sqrt(2))
        approximation=(a+b)/np.sqrt(2)
    # Signed coarse shape plus nonlinear spatially localized high-frequency energy.
    signed=np.concatenate([approximation,details[4],details[3]],axis=-1)
    energy=[]
    for d in details[:3]:
        blocks=d.reshape(*d.shape[:-1],4,-1)
        energy.append(np.sqrt(np.mean(blocks**2,axis=-1)+1e-16))
    raw=np.concatenate([signed,*energy],axis=-1).reshape(len(x),-1)
    return np.c_[raw,raw-raw[:sa.K].mean(0)] if paired else raw


class MultiscaleGP:
    def __init__(self,encoder='pca8',kernel='rbf',paired=True):
        self.encoder,self.kernel,self.paired=encoder,kernel,paired

    def feature_matrix(self,cell):
        x=self.raw_features(cell)[:,self.keep]
        return self.projection.transform(self.raw_scaler.transform(x))[:,self.latent_keep]

    def raw_features(self,cell):
        return local_features(cell,self.paired)

    def fit(self,cells,allowed):
        xx=[];yy=[];self.source_keys=[]
        for c in cells:
            ix=np.asarray(allowed[c.id],dtype=int)
            assert set(range(sa.K))<=set(ix)
            xx.append(self.raw_features(c)[ix]);yy.append(c.y[ix]-np.mean(c.y[:sa.K]))
            self.source_keys.extend((c.id,int(j)) for j in ix)
        x,y=np.concatenate(xx),np.concatenate(yy)
        assert len(y)<=1000 and np.isfinite(y).all()
        self.keep=np.flatnonzero(x.std(0)>1e-7);assert len(self.keep)>0
        self.raw_scaler=StandardScaler().fit(x[:,self.keep]);z=self.raw_scaler.transform(x[:,self.keep])
        if self.encoder=='pca8':self.projection=PCA(n_components=min(8,z.shape[1],len(z)-1),svd_solver='full').fit(z)
        elif self.encoder=='pls4':self.projection=PLSRegression(n_components=min(4,z.shape[1],len(z)-1),scale=False,max_iter=1000).fit(z,y)
        else:raise ValueError(self.encoder)
        latent=self.projection.transform(z);self.latent_keep=np.flatnonzero(latent.std(0)>1e-6)
        assert len(self.latent_keep)>0
        self.scaler=StandardScaler().fit(latent[:,self.latent_keep])
        if self.kernel=='rbf':base=RBF(1.,(1e-2,1e3))
        elif self.kernel=='matern':base=Matern(1.,(1e-2,1e3),nu=1.5)
        else:raise ValueError(self.kernel)
        self.gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(1e-3,1e3))*base+WhiteKernel(.01,(1e-6,1)),normalize_y=True,random_state=0)
        self.gp.fit(self.scaler.transform(latent[:,self.latent_keep]),y)
        return self
