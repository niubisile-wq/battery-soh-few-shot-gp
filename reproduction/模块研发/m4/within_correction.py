"""Within-source-domain conditional regression, reference-anchored inference."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from correction import features
from anchored_correction import reference_features
from source_oof import sa


def within_projection(labels, weights):
    labels = np.asarray(labels)
    h = np.eye(len(labels))
    for domain in sorted(set(labels)):
        ix = np.flatnonzero(labels == domain)
        h[np.ix_(ix, ix)] -= np.broadcast_to(weights[ix] / weights[ix].sum(), (len(ix), len(ix)))
    return h


class WithinCorrector:
    def __init__(self, view='state', learner='constant', alpha=1., clip=.05):
        self.view, self.learner, self.alpha, self.clip = view, learner, alpha, clip

    def fit(self, episodes, byid, group):
        self.source_keys = [(e['cell_id'], int(j)) for e in episodes for j in e['query_indices']]
        if self.learner == 'constant': return self
        domains = {e['domain'] for e in episodes}
        counts = {d: sum(e['domain'] == d for e in episodes) for d in domains}
        w = np.concatenate([np.full(len(e['y']), 1 / len(domains) / counts[e['domain']] / len(e['y'])) for e in episodes])
        labels = np.concatenate([np.repeat(e['domain'], len(e['y'])) for e in episodes])
        h = within_projection(labels, w)
        y = np.concatenate([np.asarray(e['y'], float) - e['predictions'][group] for e in episodes])
        x = np.concatenate([features(byid[e['cell_id']], e['predictions'][group], e['query_indices'], self.view) for e in episodes])
        self.keep = np.flatnonzero(x.std(0) > 1e-8)
        self.scaler = StandardScaler().fit(x[:, self.keep], sample_weight=w)
        z = self.scaler.transform(x[:, self.keep]); target = h @ y
        if self.learner == 'ridge':
            self.model = Ridge(alpha=self.alpha, fit_intercept=False).fit(h @ z, target, sample_weight=w * len(w))
        elif self.learner == 'rbf':
            self.z, self.gamma = z, 1 / z.shape[1]
            kernel = h @ rbf_kernel(z, z, gamma=self.gamma) @ h.T
            kernel = (kernel + kernel.T) / 2
            sw = np.sqrt(w * len(w))
            system = sw[:, None] * kernel * sw[None, :] + np.eye(len(w)) * self.alpha
            self.dual = h.T @ (sw * cho_solve(cho_factor(system, lower=True), sw * target))
        else: raise ValueError(self.learner)
        self.domain_labels, self.row_weights = labels, w
        return self

    def predict(self, cell, pred, indices=None):
        if self.learner == 'constant': return np.zeros(len(pred))
        if indices is None: indices = np.arange(sa.K, len(cell.x))
        c = sa.inference_view(cell)
        x = features(c, pred, indices, self.view)
        x0 = reference_features(c, len(pred), self.view)
        z = self.scaler.transform(x[:, self.keep]); z0 = self.scaler.transform(x0[:, self.keep])
        if self.learner == 'ridge': result = self.model.predict(z - z0)
        else: result = (rbf_kernel(z, self.z, gamma=self.gamma) - rbf_kernel(z0, self.z, gamma=self.gamma)) @ self.dual
        return np.clip(result, -self.clip, self.clip)
