"""GP with within-cell linear observations, not independent pair pseudo-labels.

For original iid measurement noise, differences have covariance sigma² D D.T.
Orthonormal cell contrasts span the same row space as adjacent differences,
with identity noise covariance. This is standard Gaussian conditioning.
This primitive does not choose encoders, source budgets or validation settings.
"""
import numpy as np
from scipy.linalg import block_diag, helmert, cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


def cell_contrasts(ids):
    ids = np.asarray(ids)
    blocks = []
    order = []
    for cid in dict.fromkeys(ids.tolist()):
        ix = np.flatnonzero(ids == cid)
        if len(ix) < 2:
            raise ValueError('Each source cell needs at least two observations')
        blocks.append(helmert(len(ix), full=False))
        order.extend(ix)
    if not blocks:
        raise ValueError('Empty source data')
    h = block_diag(*blocks)
    result = np.zeros_like(h)
    result[:, order] = h
    return result


class DifferenceGP:
    def __init__(self, kernel=None, optimize=True):
        self.kernel = kernel if kernel is not None else (
            ConstantKernel(1., (1e-3, 1e3)) * RBF(1., (1e-2, 1e3))
            + WhiteKernel(.01, (1e-6, 1.)))
        self.optimize = optimize

    def objective(self, theta, gradient=True):
        kernel = self.kernel.clone_with_theta(theta)
        if gradient:
            k, dk = kernel(self.X_train_, eval_gradient=True)
        else:
            k = kernel(self.X_train_)
        c = self.H_ @ k @ self.H_.T
        # WhiteKernel is transformed together with the latent kernel.
        cf = cho_factor(c, lower=True)
        alpha = cho_solve(cf, self.y_contrast_)
        n = len(alpha)
        loss = (.5 * self.y_contrast_ @ alpha +
                np.log(np.diag(cf[0])).sum() + .5*n*np.log(2*np.pi))
        if not gradient:
            return loss
        weight = cho_solve(cf, np.eye(n)) - np.outer(alpha, alpha)
        deriv = np.array([.5*np.sum(weight*(self.H_ @ dk[:,:,j] @ self.H_.T))
                          for j in range(dk.shape[-1])])
        return loss, deriv

    def fit(self, x, y, ids):
        self.X_train_ = np.asarray(x, dtype=float).copy()
        y = np.asarray(y, dtype=float)
        if len(x) != len(y) or len(y) != len(ids) or not np.isfinite(y).all():
            raise ValueError('Invalid observations')
        self.H_ = cell_contrasts(ids)
        contrast = self.H_ @ y
        # RMS in orthonormal contrast space; no cell-offset dependence.
        self.y_scale_ = max(float(np.sqrt(np.mean(contrast**2))), 1e-8)
        self.y_contrast_ = contrast / self.y_scale_
        theta = self.kernel.theta
        self.optimization_ = None
        if self.optimize:
            result = minimize(self.objective, theta, jac=True, method='L-BFGS-B',
                              bounds=self.kernel.bounds, options={'maxiter': 100})
            self.optimization_ = dict(success=bool(result.success),
                                      message=str(result.message), iterations=int(result.nit))
            theta = result.x
        self.kernel_ = self.kernel.clone_with_theta(theta)
        c = self.H_ @ self.kernel_(self.X_train_) @ self.H_.T
        self.factor_ = cho_factor(c, lower=True)
        self.alpha_ = cho_solve(self.factor_, self.y_contrast_)
        self.point_alpha_ = self.H_.T @ self.alpha_
        return self

    def predict(self, x):
        # Absolute level is unidentified: caller must anchor with permitted support.
        return self.kernel_(np.asarray(x), self.X_train_) @ self.point_alpha_ * self.y_scale_

    def predict_anchored(self, query, support_x, support_y):
        support_y = np.asarray(support_y, dtype=float)
        if len(support_y) != len(support_x) or not np.isfinite(support_y).all():
            raise ValueError('Invalid support')
        return self.predict(query) + np.mean(support_y - self.predict(support_x))
