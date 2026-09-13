"""Shallow residual boosting with explicit absolute/squared loss controls."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from correction import ResidualCorrector, features
from source_oof import sa


class RobustCorrector(ResidualCorrector):
    def fit(self, episodes, byid, group):
        if self.learner == 'constant':
            return super().fit(episodes, byid, group)
        if self.learner not in ('absolute', 'squared'):
            raise ValueError(self.learner)
        domains = {e['domain'] for e in episodes}
        counts = {d: sum(e['domain'] == d for e in episodes) for d in domains}
        weights = np.concatenate([np.full(len(e['y']), 1 / len(domains) / counts[e['domain']] / len(e['y'])) for e in episodes])
        target = np.concatenate([e['y'] - e['predictions'][group] for e in episodes])
        self.source_keys = [(e['cell_id'], int(j)) for e in episodes for j in e['query_indices']]
        x = np.concatenate([features(byid[e['cell_id']], e['predictions'][group], e['query_indices'], self.view) for e in episodes])
        assert np.isfinite(x).all() and np.isfinite(target).all()
        self.model = HistGradientBoostingRegressor(
            loss=self.learner + '_error', learning_rate=.05, max_iter=60,
            max_leaf_nodes=4, max_depth=2, min_samples_leaf=int(self.alpha),
            l2_regularization=10., early_stopping=False, random_state=0)
        self.model.fit(x, target, sample_weight=weights * len(weights))
        return self

    def predict(self, cell, pred, indices=None):
        if self.learner == 'constant':
            return super().predict(cell, pred, indices)
        if indices is None:
            indices = np.arange(sa.K, len(cell.x))
        x = features(sa.inference_view(cell), pred, indices, self.view)
        return np.clip(self.model.predict(x), -self.clip, self.clip)
