"""Remove source-domain mean residuals before learning conditional corrections."""
import numpy as np
from correction import ResidualCorrector


class CenteredCorrector(ResidualCorrector):
    def fit(self, episodes, byid, group):
        domains = sorted({e['domain'] for e in episodes})
        self.domain_offsets = {d: float(np.mean([
            np.mean(np.asarray(e['y'], float) - e['predictions'][group])
            for e in episodes if e['domain'] == d])) for d in domains}
        # Equal cell weighting within each domain, consistent with original loss weights.
        centered = [dict(e, y=np.asarray(e['y'], float) - self.domain_offsets[e['domain']]) for e in episodes]
        super().fit(centered, byid, group)
        assert abs(self.mean) < 1e-10
        return self
