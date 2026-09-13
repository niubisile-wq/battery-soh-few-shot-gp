from types import SimpleNamespace
import joblib
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from difference_gp import DifferenceGP
from random_function_gp import RandomFunctionGP
from random_function_branch import RandomFunctionBranch, ConditionalFunction
from test_mixed_effects_gp import fixture
from test_conditional_mixed import setup


@pytest.mark.parametrize('base', [RBF(1.), Matern(1., nu=1.5)])
def test_gradient_covariance_zero_offsets(base, tmp_path):
    x, y, ids = fixture()
    kernel = ConstantKernel(1.) * base + WhiteKernel(.01)
    m = RandomFunctionGP(kernel=kernel, optimize=False).fit(x, y, ids)
    assert np.linalg.eigvalsh(m.private_).min() > -1e-12
    np.testing.assert_array_equal(m.private_[ids[:, None] != ids[None, :]], 0)
    np.testing.assert_allclose(m.objective(m.theta_)[1], approx_derivative(lambda t: m.objective(t, False), m.theta_).ravel(), rtol=3e-5, atol=3e-6)
    zero = RandomFunctionGP(kernel=kernel, optimize=False, rho=0, learn_rho=False).fit(x, y, ids)
    reference = DifferenceGP(kernel=kernel, optimize=False).fit(x, y, ids)
    np.testing.assert_allclose(zero.predict(x), reference.predict(x), atol=1e-11, rtol=0)
    shifted = RandomFunctionGP(kernel=kernel, optimize=False).fit(x, y + np.where(ids == 'b', 7., -3.), ids)
    np.testing.assert_allclose(m.predict(x), shifted.predict(x), atol=1e-11)
    path = tmp_path / 'model.joblib'
    joblib.dump(m, path)
    np.testing.assert_array_equal(m.predict(x), joblib.load(path).predict(x))


def test_joint_target_and_boundaries(tmp_path):
    parent, c = setup()
    old = parent.gp
    # Same contrast targets; source labels may be reconstructed up to cell intercepts.
    y = old.H_.T @ old.y_contrast_ * old.y_scale_
    parent.gp = RandomFunctionGP(optimize=False).fit(old.X_train_, y, [cid for cid, _ in parent.source_keys])
    gp = parent.gp
    for private in (False, True):
        m = ConditionalFunction(parent, private)
        z = m.latent(c); zs = z[:10]; x = gp.X_train_; h = gp.H_
        d = np.diff(np.eye(10), axis=0)  # Independent nonorthogonal support basis.
        rho = gp.rho_ if private else 0.
        source = h @ (gp.kernel_(x) + gp.rho_ * gp.private_) @ h.T
        cross = h @ gp.kernel_(x, zs) @ d.T
        support = d @ (gp.kernel_(zs) + rho * gp.kernel_.k1(zs)) @ d.T
        joint = np.block([[source, cross], [cross.T, support]])
        target = np.r_[gp.y_contrast_, d @ c.y[:10] / gp.y_scale_]
        weights = np.linalg.solve(joint, target)
        query = np.c_[gp.kernel_(z, x) @ h.T, (gp.kernel_(z, zs) + rho * gp.kernel_.k1(z, zs)) @ d.T]
        pred = query @ weights * gp.y_scale_
        expected = pred[10:] + np.mean(c.y[:10] - pred[:10])
        np.testing.assert_allclose(m.predict(c), expected, atol=1e-10)
        yy = c.y.copy(); yy[10:] = np.nan
        np.testing.assert_array_equal(m.predict(c), m.predict(SimpleNamespace(**dict(c.__dict__, y=yy))))
        short = SimpleNamespace(**dict(c.__dict__, x=c.x[:13], y=c.y[:13], cycle=c.cycle[:13]))
        np.testing.assert_allclose(m.predict(c)[:3], m.predict(short), atol=1e-10)
        path = tmp_path / ('adapter_' + str(private) + '.joblib'); joblib.dump(m, path)
        np.testing.assert_array_equal(m.predict(c), joblib.load(path).predict(c))


def test_finite_optimization_and_non_exploding_private_covariance():
    x, y, ids = fixture()
    m = RandomFunctionGP().fit(x, y, ids)
    assert np.isfinite(m.predict(x)).all()
    far = np.full((1, x.shape[1]), 1e6)
    np.testing.assert_allclose(m.kernel_.k1(far, x), 0, atol=1e-12)
    assert m.kernel_.k1(far)[0, 0] == pytest.approx(m.kernel_.k1(x)[0, 0])
    with pytest.raises(ValueError):
        RandomFunctionGP().fit(x[:-1], y, ids)
