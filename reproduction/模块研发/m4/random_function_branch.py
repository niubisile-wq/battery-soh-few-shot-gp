"""Fixed audited encoder with a smooth private-function GP and K=10 support."""
from copy import deepcopy
import numpy as np
from scipy.linalg import cho_solve, helmert
from difference_branch import DifferenceBranch
from conditional_mixed import ConditionalMixed
from random_function_gp import RandomFunctionGP
from source_oof import sa


class RandomFunctionBranch(DifferenceBranch):
    @classmethod
    def from_encoder(cls, reference, cells, allowed, optimize=True):
        assert reference.target == 'difference'
        result = cls(target='difference', kernel=reference.kernel, optimize=optimize)
        for attr in ('keep', 'raw_scaler', 'pca', 'latent_keep', 'scaler', 'source_keys', 'scale'):
            setattr(result, attr, deepcopy(getattr(reference, attr)))
        byid = {c.id: c for c in cells}
        assert set(byid) == set(allowed)
        assert set(result.source_keys) == {(cid, int(j)) for cid, ix in allowed.items() for j in ix}
        assert len(result.source_keys) == sum(map(len, allowed.values())) <= 1000
        assert all(set(range(sa.K)) <= set(ix) for ix in allowed.values())
        y = np.array([byid[cid].y[j] - np.mean(byid[cid].y[:sa.K]) for cid, j in result.source_keys])
        result.gp = RandomFunctionGP(kernel=deepcopy(reference.gp.kernel), optimize=optimize).fit(
            reference.gp.X_train_, y, [cid for cid, _ in result.source_keys])
        return result


class ConditionalFunction(ConditionalMixed):
    def predict(self, c):
        z = self.latent(c)
        gp = self.parent.gp
        zs, ys = z[:sa.K], np.asarray(c.y[:sa.K], float)
        assert len(ys) == sa.K and np.isfinite(ys).all()
        h = helmert(sa.K, full=False)
        rho = gp.rho_ if self.private else 0.
        cross_s = gp.kernel_(zs, gp.X_train_) @ gp.H_.T
        solved = cho_solve(gp.factor_, cross_s.T)
        latent = gp.kernel_(zs, zs) - cross_s @ solved + rho * gp.kernel_.k1(zs, zs)
        observed = latent + gp.kernel_.k2(zs)
        cov = h @ observed @ h.T
        mean = gp.predict(zs) / gp.y_scale_
        weights = np.linalg.solve((cov + cov.T) / 2, h @ (ys / gp.y_scale_ - mean))
        anchor = np.mean(ys) / gp.y_scale_ - np.mean(mean + latent @ h.T @ weights)
        out = []
        for j in range(sa.K, len(z), 256):
            zz = z[j:j + 256]
            cross = gp.kernel_(zz, gp.X_train_) @ gp.H_.T
            cov_qs = gp.kernel_(zz, zs) - cross @ solved + rho * gp.kernel_.k1(zz, zs)
            out.append((gp.predict(zz) / gp.y_scale_ + cov_qs @ h.T @ weights + anchor) * gp.y_scale_)
        prediction = np.concatenate(out)
        assert np.isfinite(prediction).all()
        return prediction
