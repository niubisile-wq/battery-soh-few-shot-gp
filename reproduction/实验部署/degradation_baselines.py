"""Empirical degradation fit using only observed support cycle coordinates."""
import numpy as np


def exponential_trend(support_cycles, support_soh, query_cycles):
    """Fit log(SOH) = intercept + slope * cycle; no query labels needed.

    Positive slopes are allowed so capacity regeneration is not deleted.
    Invalid inputs fail explicitly rather than silently changing the model.
    """
    t = np.asarray(support_cycles, dtype=float)
    y = np.asarray(support_soh, dtype=float)
    q = np.asarray(query_cycles, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 3:
        raise ValueError("Exponential fit requires at least three paired support points")
    if not (np.isfinite(t).all() and np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError("Nonfinite cycle or SOH")
    if (y <= 0).any() or (np.diff(t) <= 0).any():
        raise ValueError("Positive SOH and strictly increasing cycles required")
    scale = t[-1] - t[0]
    slope, intercept = np.polyfit((t - t[0]) / scale, np.log(y), 1)
    with np.errstate(over="raise", invalid="raise"):
        return np.exp(intercept + slope * (q - t[0]) / scale)
