"""Reusable frozen M4 interface; effective_gain already includes global shrinkage.

No fitting, tuning, model selection or dataset-specific rule is performed here.
The caller must supply a frozen parent and the audited source branch.
"""
import numpy as np
from source_oof import sa
from random_function_branch import ConditionalFunction


class RandomFunctionM4:
    def __init__(self, parent, source_branch, effective_gain, parent_mode=None):
        if not np.isfinite(effective_gain) or not 0 <= effective_gain <= 1:
            raise ValueError('Effective gain must be finite and in [0,1]')
        self.parent = parent
        self.source_branch = source_branch
        self.effective_gain = float(effective_gain)
        self.parent_mode = parent_mode

    def predict_from_parent(self, cell, parent_prediction):
        c = sa.inference_view(cell)
        p = np.asarray(parent_prediction, dtype=float)
        if p.shape != (len(c.x) - sa.K,) or not np.isfinite(p).all():
            raise ValueError('Invalid frozen parent prediction')
        if self.effective_gain == 0: return p.copy()
        q = np.asarray(ConditionalFunction(self.source_branch, True).predict(c), dtype=float)
        if q.shape != p.shape or not np.isfinite(q).all():
            raise ValueError('Invalid private-function prediction')
        return p + self.effective_gain * (q - p)

    def predict(self, cell):
        c = sa.inference_view(cell)
        p = self.parent.predict(c) if self.parent_mode is None else self.parent.predict(c, self.parent_mode)
        return self.predict_from_parent(c, p)
