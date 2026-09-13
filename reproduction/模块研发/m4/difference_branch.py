"""Budget-only shared PCA representation for direct versus contrast GP training."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from multiscale import local_features
from difference_gp import DifferenceGP, cell_contrasts
from source_oof import sa


class DifferenceBranch:
    def __init__(self, target='difference', kernel='rbf', optimize=True):
        assert target in ('difference','direct') and kernel in ('rbf','matern')
        self.target,self.kernel,self.optimize=target,kernel,optimize

    def fit(self,cells,allowed):
        xx=[];yy=[];ids=[];self.source_keys=[]
        for c in cells:
            ix=np.asarray(sorted(allowed[c.id]),dtype=int)
            assert len(ix)==len(set(ix)) and set(range(sa.K))<=set(ix)
            xx.append(local_features(c)[ix])
            yy.append(c.y[ix]-np.mean(c.y[:sa.K]))
            ids.extend([c.id]*len(ix));self.source_keys.extend((c.id,int(i)) for i in ix)
        x,y=np.concatenate(xx),np.concatenate(yy)
        assert len(y)<=1000 and np.isfinite(y).all()
        self.keep=np.flatnonzero(x.std(0)>1e-7)
        self.raw_scaler=StandardScaler().fit(x[:,self.keep])
        raw=self.raw_scaler.transform(x[:,self.keep])
        self.pca=PCA(n_components=min(8,len(y)-1,len(self.keep)),svd_solver='full').fit(raw)
        latent=self.pca.transform(raw);self.latent_keep=np.flatnonzero(latent.std(0)>1e-6)
        self.scaler=StandardScaler().fit(latent[:,self.latent_keep]);z=self.scaler.transform(latent[:,self.latent_keep])
        base=RBF(1.,(1e-2,1e3)) if self.kernel=='rbf' else Matern(1.,(1e-2,1e3),nu=1.5)
        kernel=ConstantKernel(1.,(1e-3,1e3))*base+WhiteKernel(.01,(1e-6,1.))
        contrast=cell_contrasts(ids)@y
        self.scale=max(float(np.sqrt(np.mean(contrast**2))),1e-8)
        if self.target=='difference':
            self.gp=DifferenceGP(kernel,self.optimize).fit(z,y,ids)
        else:
            self.gp=GaussianProcessRegressor(kernel=kernel,normalize_y=False,alpha=0.,
                optimizer='fmin_l_bfgs_b' if self.optimize else None,random_state=0).fit(z,y/self.scale)
        return self

    def predict(self,c):
        raw=self.raw_scaler.transform(local_features(c)[:,self.keep])
        z=self.scaler.transform(self.pca.transform(raw)[:,self.latent_keep])
        p=self.gp.predict(z)
        if self.target=='direct':p=p*self.scale
        assert np.isfinite(c.y[:sa.K]).all()
        return p[sa.K:]+np.mean(c.y[:sa.K])-np.mean(p[:sa.K])
