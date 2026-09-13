"""Separate signed-shape and local-energy encoders, joint/additive GP control."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF,WhiteKernel
from multiscale import local_features
from source_oof import sa
from gp import Columns


def column_groups():
    shape=[offset+channel*28+j for offset in (0,84) for channel in range(3) for j in range(16)]
    energy=[offset+channel*28+j for offset in (0,84) for channel in range(3) for j in range(16,28)]
    return [np.array(shape),np.array(energy)]


class StructuredMultiscaleGP:
    def __init__(self,encoder='pca8',kernel='additive'):
        self.encoder,self.kernel=encoder,kernel

    def feature_matrix(self,cell):
        raw=local_features(cell);blocks=[]
        for spec in self.blocks:
            z=spec['scaler'].transform(raw[:,spec['columns']])
            blocks.append(spec['projection'].transform(z)[:,spec['keep']])
        return np.concatenate(blocks,axis=1)

    def fit(self,cells,allowed):
        xx=[];yy=[];self.source_keys=[]
        for c in cells:
            ix=np.asarray(allowed[c.id],int);assert set(range(sa.K))<=set(ix)
            xx.append(local_features(c)[ix]);yy.append(c.y[ix]-np.mean(c.y[:sa.K]))
            self.source_keys.extend((c.id,int(j)) for j in ix)
        x,y=np.concatenate(xx),np.concatenate(yy);assert len(y)<=1000 and np.isfinite(y).all()
        self.blocks=[];latent=[]
        for columns in column_groups():
            columns=columns[x[:,columns].std(0)>1e-7];assert len(columns)>0
            scaler=StandardScaler().fit(x[:,columns]);z=scaler.transform(x[:,columns])
            if self.encoder=='pca8':projection=PCA(n_components=min(4,z.shape[1],len(z)-1),svd_solver='full').fit(z)
            elif self.encoder=='pls4':projection=PLSRegression(n_components=min(2,z.shape[1],len(z)-1),scale=False,max_iter=1000).fit(z,y)
            else:raise ValueError(self.encoder)
            p=projection.transform(z);keep=np.flatnonzero(p.std(0)>1e-6);assert len(keep)>0
            self.blocks.append(dict(columns=columns,scaler=scaler,projection=projection,keep=keep));latent.append(p[:,keep])
        z=np.concatenate(latent,axis=1);self.scaler=StandardScaler().fit(z)
        if self.kernel=='joint':signal=ConstantKernel(1.,(1e-3,1e3))*RBF(1.,(1e-2,1e3))
        elif self.kernel=='additive':
            split=latent[0].shape[1]
            signal=Columns(ConstantKernel(1.,(1e-3,1e3))*RBF(1.,(1e-2,1e3)),tuple(range(split)))+Columns(ConstantKernel(1.,(1e-3,1e3))*RBF(1.,(1e-2,1e3)),tuple(range(split,z.shape[1])))
        else:raise ValueError(self.kernel)
        self.gp=GaussianProcessRegressor(kernel=signal+WhiteKernel(.01,(1e-6,1)),normalize_y=True,random_state=0)
        self.gp.fit(self.scaler.transform(z),y);return self
