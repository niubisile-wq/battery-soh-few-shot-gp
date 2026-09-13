"""Source-domain deletion ensemble with an input-dependent disagreement gate.

No reliability labels are fitted. Deletion applies to correction fitting, not a
fresh nesting of the inherited complete-parent OOF generator.
"""
import numpy as np
from correction import ResidualCorrector


class ConsensusCorrector(ResidualCorrector):
    def fit(self, episodes, byid, group):
        self.source_keys = [(e['cell_id'], int(j)) for e in episodes for j in e['query_indices']]
        if self.learner == 'constant':
            return super().fit(episodes, byid, group)
        domains = sorted({e['domain'] for e in episodes})
        if len(domains) < 2:
            raise ValueError('At least two source domains are required')
        self.members = []
        self.exclusions = []
        for held in domains:
            training = [e for e in episodes if e['domain'] != held]
            ids = {e['cell_id'] for e in training}
            model = ResidualCorrector(self.view, self.learner, self.alpha, self.clip)
            model.fit(training, {k: byid[k] for k in ids}, group)
            self.members.append(model)
            self.exclusions.append(dict(held_domain=held, training_ids=sorted(ids)))
        return self

    def components(self, cell, pred, indices=None):
        values = np.stack([m.predict(cell, pred, indices) for m in self.members])
        mean = values.mean(axis=0)
        magnitude = np.abs(values).mean(axis=0)
        gate = np.divide(np.abs(mean), magnitude, out=np.zeros_like(mean), where=magnitude > 1e-12)
        return mean, np.clip(gate, 0, 1)

    def predict(self, cell, pred, indices=None):
        if self.learner == 'constant':
            return super().predict(cell, pred, indices)
        mean, gate = self.components(cell, pred, indices)
        return np.clip(mean * gate, -self.clip, self.clip)
