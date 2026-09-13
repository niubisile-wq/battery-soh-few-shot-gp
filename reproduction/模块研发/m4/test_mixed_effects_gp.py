import numpy as np
import joblib
from scipy.optimize._numdiff import approx_derivative
from mixed_effects_gp import MixedEffectsGP,slope_covariance
from difference_gp import DifferenceGP


def fixture():
    rng=np.random.default_rng(67);x=rng.normal(size=(18,3))
    ids=np.array(['a','b','c']*6)
    y=np.sin(x[:,0])+.2*x[:,1]+np.array([{'a':2.,'b':-1.,'c':.5}[i] for i in ids])
    return x,y,ids


def test_block_psd_and_zero_equivalence():
    x,y,ids=fixture();s=slope_covariance(x,ids)
    assert np.linalg.eigvalsh(s).min()>-1e-12
    np.testing.assert_array_equal(s[ids[:,None]!=ids[None,:]],0)
    a=DifferenceGP(optimize=False).fit(x,y,ids)
    b=MixedEffectsGP(optimize=False,rho=0,learn_rho=False).fit(x,y,ids)
    np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-12,rtol=0)
    np.testing.assert_allclose(a.objective(a.kernel.theta)[1],b.objective(b.theta_)[1],atol=1e-11)


def test_gradient_dense_solution_offsets_and_serialization(tmp_path):
    x,y,ids=fixture();m=MixedEffectsGP(optimize=False).fit(x,y,ids)
    analytic=m.objective(m.theta_)[1]
    numeric=approx_derivative(lambda t:m.objective(t,False),m.theta_).ravel()
    np.testing.assert_allclose(analytic,numeric,rtol=2e-5,atol=2e-6)
    h=m.H_;c=h@(m.kernel_(x)+m.rho_*slope_covariance(x,ids))@h.T
    expected=m.kernel_(x,x)@h.T@np.linalg.solve(c,h@y)
    np.testing.assert_allclose(expected,m.predict(x),atol=1e-11)
    shifted=MixedEffectsGP(optimize=False).fit(x,y+np.where(ids=='b',7.,-3.),ids)
    np.testing.assert_allclose(shifted.predict(x),m.predict(x),atol=1e-11)
    path=tmp_path/'m.joblib';joblib.dump(m,path)
    np.testing.assert_array_equal(m.predict(x),joblib.load(path).predict(x))


def test_optimized_and_invalid():
    import pytest
    x,y,ids=fixture();m=MixedEffectsGP().fit(x,y,ids)
    assert np.isfinite(m.predict(x)).all() and 1e-6<=m.rho_<=1e3
    with pytest.raises(ValueError):MixedEffectsGP(rho=-1)
    with pytest.raises(ValueError):slope_covariance(x[:-1],ids)
