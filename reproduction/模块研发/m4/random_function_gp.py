"""Shared GP plus independent smooth cell functions, in contrast space.

K_source = K_shared + rho * same_cell * K_signal + noise.
Private and shared functions share length scales; rho is a variance ratio.
This is a standard covariance construction, not a novelty claim.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from mixed_effects_gp import MixedEffectsGP
from difference_gp import cell_contrasts


class RandomFunctionGP(MixedEffectsGP):
    def covariance(self, kernel, gradient=False):
        x = self.X_train_
        if gradient:
            k, dk = kernel(x, eval_gradient=True)
            signal, ds = kernel.k1(x, eval_gradient=True)
            assert dk.shape[-1] == ds.shape[-1] + 1
            dp = np.zeros_like(dk)
            dp[:, :, :ds.shape[-1]] = ds * self.cell_mask_[:, :, None]
            return k, signal * self.cell_mask_, dk, dp
        return kernel(x), kernel.k1(x) * self.cell_mask_

    def objective(self, theta, gradient=True):
        kt, rho = self.unpack(theta)
        kernel = self.kernel.clone_with_theta(kt)
        values = self.covariance(kernel, gradient)
        k, private = values[:2]
        h = self.H_
        c = h @ (k + rho * private) @ h.T
        factor = cho_factor(c, lower=True)
        alpha = cho_solve(factor, self.y_contrast_)
        n = len(alpha)
        loss = .5 * self.y_contrast_ @ alpha + np.log(np.diag(factor[0])).sum() + .5 * n * np.log(2 * np.pi)
        if not gradient:
            return loss
        weight = cho_solve(factor, np.eye(n)) - np.outer(alpha, alpha)
        dk, dp = values[2:]
        derivatives = [.5 * np.sum(weight * (h @ (dk[:, :, j] + rho * dp[:, :, j]) @ h.T)) for j in range(dk.shape[-1])]
        if self.learn_rho:
            derivatives.append(.5 * np.sum(weight * (h @ (rho * private) @ h.T)))
        return loss, np.asarray(derivatives)

    def fit(self, x, y, ids):
        x, y, ids = np.asarray(x, float), np.asarray(y, float), np.asarray(ids)
        if x.ndim != 2 or not x.shape[1] or y.ndim != 1 or len(x) != len(y) or len(ids) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid random-function observations')
        self.X_train_ = x.copy()
        self.cell_mask_ = ids[:, None] == ids[None, :]
        self.H_ = cell_contrasts(ids)
        contrast = self.H_ @ y
        self.y_scale_ = max(float(np.sqrt(np.mean(contrast ** 2))), 1e-8)
        self.y_contrast_ = contrast / self.y_scale_
        theta, bounds = self.kernel.theta, self.kernel.bounds
        if self.learn_rho:
            theta = np.r_[theta, np.log(self.rho)]
            bounds = np.vstack([bounds, np.log([1e-6, 1e3])])
        self.optimization_ = None
        if self.optimize:
            result = minimize(self.objective, theta, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            theta = result.x
            self.optimization_ = dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit))
        kt, self.rho_ = self.unpack(theta)
        self.theta_ = theta
        self.kernel_ = self.kernel.clone_with_theta(kt)
        k, self.private_ = self.covariance(self.kernel_)
        c = self.H_ @ (k + self.rho_ * self.private_) @ self.H_.T
        self.factor_ = cho_factor(c, lower=True)
        self.alpha_ = cho_solve(self.factor_, self.y_contrast_)
        self.point_alpha_ = self.H_.T @ self.alpha_
        self.diagnostics_ = dict(rho=self.rho_, rho_bound_hit=bool(self.learn_rho and (self.rho_ <= 1.0001e-6 or self.rho_ >= 999.9)),
                                 shared_kernel=str(self.kernel_), random_contrast_trace=float(self.rho_ * np.trace(self.H_ @ self.private_ @ self.H_.T)))
        return self
