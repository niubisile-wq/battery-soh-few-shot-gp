import unittest
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize._numdiff import approx_derivative
from difference_gp import DifferenceGP, cell_contrasts


class DifferenceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 2, 12)[:,None]
        self.ids = ['a']*5 + ['b']*7
        self.y = np.sin(self.x[:,0]) + np.r_[np.ones(5), np.full(7,3.)]

    def test_contrasts_and_offsets(self):
        h = cell_contrasts(self.ids)
        np.testing.assert_allclose(h @ h.T, np.eye(10), atol=1e-14)
        np.testing.assert_allclose(h @ np.ones(12), 0, atol=1e-14)
        a = DifferenceGP(optimize=False).fit(self.x, self.y, self.ids)
        b = DifferenceGP(optimize=False).fit(self.x, self.y + np.r_[np.full(5,10),np.full(7,-2)], self.ids)
        np.testing.assert_allclose(a.predict(self.x), b.predict(self.x), atol=1e-12)

    def test_adjacent_correlated_noise_equivalence(self):
        model = DifferenceGP(optimize=False).fit(self.x, self.y, self.ids)
        d = block_diag(np.diff(np.eye(5),axis=0), np.diff(np.eye(7),axis=0))
        # Adjacent differences are correlated, off-diagonal noise is not zero.
        self.assertGreater(np.abs(d@d.T-np.diag(np.diag(d@d.T))).sum(), 0)
        k = model.kernel_(self.x)
        pred = model.kernel_(self.x,self.x) @ d.T @ np.linalg.solve(d@k@d.T,d@self.y)
        np.testing.assert_allclose(pred, model.predict(self.x), atol=1e-11)

    def test_gradient_and_anchor(self):
        model = DifferenceGP(optimize=False).fit(self.x, self.y, self.ids)
        theta = model.kernel.theta
        analytic = model.objective(theta)[1]
        numeric = approx_derivative(lambda t:model.objective(t,False),theta).ravel()
        np.testing.assert_allclose(analytic,numeric,rtol=1e-5,atol=1e-6)
        p = model.predict_anchored(self.x[:3],self.x[:3],self.y[:3])
        self.assertAlmostEqual(float(p.mean()),float(self.y[:3].mean()),12)
        fitted = DifferenceGP().fit(self.x,self.y,self.ids)
        self.assertTrue(np.isfinite(fitted.predict(self.x)).all())


if __name__ == '__main__':
    unittest.main()
