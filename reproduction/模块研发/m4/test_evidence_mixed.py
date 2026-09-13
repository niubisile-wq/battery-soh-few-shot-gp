from types import SimpleNamespace
import numpy as np
from scipy.linalg import helmert,cho_solve
from scipy.stats import multivariate_normal
from scipy.special import expit
from test_conditional_mixed import setup
from evidence_mixed import EvidenceMixed
from conditional_mixed import ConditionalMixed


def test_likelihood_and_weight_boundaries():
    model,c=setup();adapter=EvidenceMixed(model);w=adapter.weights(c)
    z=ConditionalMixed(model).latent(c)[:10];gp=model.gp;h=helmert(10,full=False)
    cross=gp.kernel_(z,gp.X_train_)@gp.H_.T
    s=gp.kernel_(z)-cross@cho_solve(gp.factor_,cross.T)
    residual=h@(c.y[:10]/gp.y_scale_-gp.predict(z)/gp.y_scale_)
    logs=[multivariate_normal.logpdf(residual,mean=np.zeros(9),cov=h@cov@h.T)
          for cov in (s,s+gp.rho_*(z@z.T)/z.shape[1])]
    np.testing.assert_allclose(w['private'],expit(logs[1]-logs[0]),atol=1e-10)
    assert 0<=w['private']<=1 and abs(w['private']+w['shared']-1)<1e-12


def test_no_query_labels_or_future_inputs():
    model,c=setup();m=EvidenceMixed(model);p=m.predict(c);yy=c.y.copy();yy[10:]=999
    altered=SimpleNamespace(**dict(c.__dict__,y=yy));assert m.weights(c)==m.weights(altered)
    np.testing.assert_allclose(p,m.predict(altered),atol=1e-12)
    short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
    assert m.weights(c)==m.weights(short)
    np.testing.assert_allclose(p[:3],m.predict(short),atol=1e-10)
    uniform=EvidenceMixed(model,'uniform');assert uniform.weights(c)['private']==.5
    np.testing.assert_allclose(uniform.predict(c),.5*(ConditionalMixed(model,True).predict(c)+ConditionalMixed(model,False).predict(c)))
