from types import SimpleNamespace
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize_scalar
from test_conditional_mixed import setup
from conditional_mixed import ConditionalMixed
from support_cv_mixed import SupportCVMixed,prefix_prediction


def test_independent_adjacent_difference_prediction_and_no_held_label():
    model,c=setup();gp=model.gp;z=ConditionalMixed(model).latent(c)[:10];y=c.y[:10].astype(float)
    for t in range(5,10):
        for private in (False,True):
            zs=z[:t];zz=z[t:t+1];d=np.diff(np.eye(t),axis=0);rho=gp.rho_ if private else 0
            k=gp.kernel_(zs,gp.X_train_)@gp.H_.T
            solved=cho_solve(gp.factor_,k.T)
            latent=gp.kernel_(zs,zs)-k@solved+rho*zs@zs.T/z.shape[1]
            covariance=latent+np.eye(t)*gp.kernel_.k2.noise_level
            mean=gp.predict(zs)/gp.y_scale_
            alpha=np.linalg.solve(d@covariance@d.T,d@(y[:t]/gp.y_scale_-mean))
            cross=gp.kernel_(zz,zs)-(gp.kernel_(zz,gp.X_train_)@gp.H_.T)@solved+rho*zz@zs.T/z.shape[1]
            expected=(gp.predict(zz)/gp.y_scale_+cross@d.T@alpha+np.mean(y[:t])/gp.y_scale_-np.mean(mean+latent@d.T@alpha))[0]*gp.y_scale_
            actual=prefix_prediction(gp,z,y,t,private)
            np.testing.assert_allclose(actual,expected,atol=1e-10)
            changed=y.copy();changed[t:]=999
            assert actual==prefix_prediction(gp,z,changed,t,private)


def test_weight_solves_penalized_objective_and_boundaries():
    model,c=setup();m=SupportCVMixed(model);w=m.weights(c)
    z=ConditionalMixed(model).latent(c)[:10];y=c.y[:10].astype(float)
    p=np.array([prefix_prediction(model.gp,z,y,t,False) for t in range(5,10)])
    q=np.array([prefix_prediction(model.gp,z,y,t,True) for t in range(5,10)])
    objective=lambda a:np.sum((y[5:]-(1-a)*p-a*q)**2)+w['penalty']*(a-.5)**2
    result=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-12})
    assert objective(w['private'])<=result.fun+1e-12
    assert 0<=w['private']<=1
    yy=c.y.copy();yy[10:]=999
    changed=SimpleNamespace(**dict(c.__dict__,y=yy))
    assert w==m.weights(changed)
    np.testing.assert_allclose(m.predict(c),m.predict(changed),atol=1e-12)
    short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
    assert w==m.weights(short)
    np.testing.assert_allclose(m.predict(c)[:3],m.predict(short),atol=1e-10)
    uniform=SupportCVMixed(model,'uniform');assert uniform.weights(c)['private']==.5
    np.testing.assert_allclose(uniform.predict(c),.5*(ConditionalMixed(model,False).predict(c)+ConditionalMixed(model,True).predict(c)))
