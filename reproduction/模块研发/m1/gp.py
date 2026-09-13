"""Exact GP controls with supported query-causal posterior adaptation."""
import warnings
import numpy as np
from scipy.linalg import cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel,ConstantKernel,RBF,Matern,WhiteKernel
from sklearn.linear_model import Ridge
from data import K
from features import FeatureMap,target_shift


class Columns(Kernel):
    def __init__(self,kernel,columns):
        self.kernel=kernel;self.columns=columns
    @property
    def hyperparameters(self):return self.kernel.hyperparameters
    @property
    def theta(self):return self.kernel.theta
    @theta.setter
    def theta(self,value):self.kernel.theta=value
    @property
    def bounds(self):return self.kernel.bounds
    def __call__(self,X,Y=None,eval_gradient=False):
        return self.kernel(X[:,self.columns],None if Y is None else Y[:,self.columns],eval_gradient=eval_gradient)
    def diag(self,X):return self.kernel.diag(X[:,self.columns])
    def is_stationary(self):return self.kernel.is_stationary()
    def __repr__(self):return f'Columns({self.columns},{self.kernel})'


def make_kernel(kind,init,dimension,health_components=None):
    if kind=='group':
        if dimension!=21:raise ValueError('Grouped control requires original 21 HI')
        groups=[Columns(ConstantKernel(1.,(1e-3,1e3))*RBF(init,(1e-2,1e3)),tuple(range(a,a+7))) for a in [0,7,14]]
        signal=groups[0]+groups[1]+groups[2]
    elif kind=='health_additive':
        if health_components is None or not 0 < health_components < dimension:
            raise ValueError('Health additive kernel requires raw and health groups')
        split=dimension-health_components
        raw=Columns(ConstantKernel(1.,(1e-3,1e3))*RBF(init,(1e-2,1e3)),tuple(range(split)))
        health=Columns(ConstantKernel(1.,(1e-3,1e3))*RBF(init,(1e-2,1e3)),tuple(range(split,dimension)))
        signal=raw+health
    else:
        length=np.full(dimension,init) if kind=='ard' else init
        base=Matern(length,(1e-2,1e3),nu=1.5) if kind=='matern' else RBF(length,(1e-2,1e3))
        signal=ConstantKernel(1.,(1e-3,1e3))*base
    return signal+WhiteKernel(.001,(1e-6,.1))


class GPModel:
    def __init__(self,spec,config):
        self.spec=spec;self.config=config;self.mean_model=None

    def fit(self,cells,indices):
        self.features=FeatureMap(self.spec,self.config).fit(cells,indices)
        z=np.concatenate([self.features.transform(c)[indices[c.id]] for c in cells])
        y=np.concatenate([c.y[indices[c.id]]-target_shift(c,self.spec.get('anchored',False)) for c in cells])
        # Original GPR receives float32 labels just as in the champion runner.
        if self.spec.get('mean')=='ridge':
            self.mean_model=Ridge(alpha=self.config['mean_alpha']).fit(z,y)
            y=y-self.mean_model.predict(z)
        self.gp=GaussianProcessRegressor(kernel=make_kernel(self.spec['kernel'],self.config.get('init',1.),z.shape[1],
                                                            self.config.get('components')),
                                        normalize_y=True,n_restarts_optimizer=0,random_state=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always');self.gp.fit(z,y)
        self.warnings=sorted({str(t.message) for t in w})
        return self

    def predict(self,cell,mode):
        z=self.features.transform(cell)
        p=self.gp.predict(z)
        if self.mean_model is not None:p=p+self.mean_model.predict(z)
        p=p+target_shift(cell,self.spec.get('anchored',False))
        if mode=='source_only':return p[K:]
        residual=cell.y[:K]-p[:K]
        if mode=='source_bias':return p[K:]+residual.mean()
        if mode!='posterior':raise ValueError(mode)
        zs,zq=z[:K],z[K:]
        gp=self.gp
        # Conditional covariance uses only the first K target observations.
        ts=gp.kernel_(gp.X_train_,zs)
        solved=cho_solve((gp.L_,True),ts,check_finite=False)
        ss=gp.kernel_(zs)-ts.T@solved
        ss=.5*(ss+ss.T)+np.eye(K)*1e-9
        qs=gp.kernel_(zq,zs)-gp.kernel_(zq,gp.X_train_)@solved
        return p[K:]+qs@np.linalg.solve(ss,residual)

    def serialize(self):
        return {'kernel':str(self.gp.kernel_),'kernel_theta':self.gp.kernel_.theta.tolist(),
                'log_marginal_likelihood':float(self.gp.log_marginal_likelihood_value_),
                'features':self.features.serialize(),'warnings':self.warnings}
