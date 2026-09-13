"""Source-trained reference-relative partial-charge geometry branch."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

K = 10


def features(cell, view):
    x = np.asarray(cell.x, dtype=float)
    assert len(x) >= K and np.isfinite(x).all()
    v, current, t = x[:, 0], x[:, 1], x[:, 2]
    dt = np.diff(t, axis=1)
    assert np.min(dt) >= -1e-6
    dq = np.maximum(.5*(current[:, 1:]+current[:, :-1])*np.maximum(dt, 0)/3600, 0)
    q = np.c_[np.zeros(len(x)), np.cumsum(dq, axis=1)]
    assert np.all(q[:, -1] > 0)
    bins = []
    for vi, qi in zip(v, q):
        envelope = np.maximum.accumulate(vi)
        u, idx = np.unique(envelope, return_index=True)
        assert len(u) >= 2
        cumulative = np.interp(np.linspace(u[0], u[-1], 9), u, qi[idx])
        bins.append(np.maximum(np.diff(cumulative), qi[-1]*1e-6))
    raw = np.c_[bins, q[:, -1], np.maximum(t[:, -1]-t[:, 0], 1e-9),
                np.maximum(current.mean(axis=1), 1e-9)]
    relative = np.log(np.maximum(raw, 1e-12)/np.maximum(raw[:K].mean(axis=0), 1e-12))
    if view == 'relative':
        result = relative
    elif view == 'relative_absolute':
        absolute = np.log(np.maximum(raw[:, :9]/cell.reference_capacity, 1e-12))
        result = np.c_[relative, absolute, v[:, 0], v[:, -1]-v[:, 0],
                       current.std(axis=1)/np.maximum(abs(current.mean(axis=1)), 1e-9)]
    else:
        raise ValueError(view)
    assert np.isfinite(result).all()
    return result


class GeometryGP:
    def __init__(self, view, kernel):
        self.view, self.kernel = view, kernel

    def fit(self, cells, allowed):
        xx, yy = [], []
        self.source_keys = []
        for c in cells:
            idx = np.asarray(allowed[c.id], dtype=int)
            assert set(range(K)) <= set(idx)
            xx.append(features(c, self.view)[idx])
            yy.append(c.y[idx]-np.mean(c.y[:K]))
            self.source_keys.extend((c.id, int(j)) for j in idx)
        x, y = np.concatenate(xx), np.concatenate(yy)
        assert np.isfinite(y).all() and len(y) <= 1000
        self.scaler = StandardScaler().fit(x)
        base = RBF(1., (1e-2, 1e3)) if self.kernel == 'rbf' else Matern(1., (1e-2, 1e3), nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=ConstantKernel(1., (1e-3, 1e3))*base+
            WhiteKernel(.01, (1e-6, 1)), normalize_y=True, random_state=0)
        self.gp.fit(self.scaler.transform(x), y)
        return self

    def predict(self, cell):
        x = self.scaler.transform(features(cell, self.view))
        p = np.concatenate([self.gp.predict(x[j:j+256]) for j in range(0, len(x), 256)])
        return p[K:] + float(np.mean(cell.y[:K])) - float(np.mean(p[:K]))
