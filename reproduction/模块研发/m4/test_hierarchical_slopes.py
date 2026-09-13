import numpy as np
from scipy.optimize._numdiff import approx_derivative
from hierarchical_slopes import HierarchicalSlopesGP
from mixed_effects_gp import MixedEffectsGP


def fixture():
    rng=np.random.default_rng(127);x=rng.normal(size=(24,3))
    cells=np.repeat(['a','b','c','d'],6);domains=np.repeat(['u','v'],12)
    y=np.sin(x[:,0])+.2*x[:,1]
    return x,y,cells,domains


def test_zero_domain_equivalence_and_nested_covariance():
    x,y,c,d=fixture();m=HierarchicalSlopesGP(optimize=False,rhos=(.1,0),learn_rhos=False).fit(x,y,c,d)
    old=MixedEffectsGP(optimize=False,rho=.1,learn_rho=False).fit(x,y,c)
    np.testing.assert_allclose(m.predict(x),old.predict(x),atol=1e-12)
    assert abs(m.nuisance_[0][0,7])==0 and abs(m.nuisance_[1][0,7])>0
    assert m.nuisance_[1][0,13]==0
    assert np.linalg.eigvalsh(m.nuisance_[1]).min()>-1e-12


def test_gradient_and_source_dense_solve():
    x,y,c,d=fixture();m=HierarchicalSlopesGP(optimize=False).fit(x,y,c,d)
    grad=m.objective(m.theta_)[1];numerical=approx_derivative(lambda t:m.objective(t,False),m.theta_).ravel()
    np.testing.assert_allclose(grad,numerical,rtol=2e-5,atol=2e-6)
    cov=m.H_@(m.kernel_(x)+m.rho_cell_*m.nuisance_[0]+m.rho_domain_*m.nuisance_[1])@m.H_.T
    p=m.kernel_(x,x)@m.H_.T@np.linalg.solve(cov,m.H_@y)
    np.testing.assert_allclose(p,m.predict(x),atol=1e-11)
    assert m.rho_==m.rho_cell_+m.rho_domain_


def test_optimization_and_invalid_nested_ids():
    import pytest
    x,y,c,d=fixture();m=HierarchicalSlopesGP().fit(x,y,c,d)
    assert np.isfinite(m.predict(x)).all()
    bad=d.copy();bad[0]='v'
    with pytest.raises(ValueError):HierarchicalSlopesGP().fit(x,y,c,bad)
