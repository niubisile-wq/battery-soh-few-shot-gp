"""Candidate extraction for a versioned partial-charge protocol.

Window endpoints must be selected on development data before use. This module
does not select them using target trajectories or labels. It preserves elapsed
seconds, capacity in Ah and original cycle numbers for subsequent auditing.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Window:
    low_v: float
    high_v: float
    points: int = 128
    min_observations: int = 8

    def __post_init__(self):
        if not (np.isfinite(self.low_v) and np.isfinite(self.high_v)
                and self.low_v < self.high_v and self.points >= 2
                and self.min_observations >= 2):
            raise ValueError("Invalid fixed voltage window")


def extract(cycle, window):
    """Return a single contiguous observed voltage window, or fail explicitly.

    Positive-current runs are never joined across discharge or rests. Select
    the first complete upward traversal; voltage rebound later in the cycle
    does not extend the window. Interpolation uses measured time coordinates.
    """
    arrays = [np.asarray(cycle[k], dtype=float) for k in
              ("voltage_in_V", "current_in_A", "time_in_s")]
    v, i, t = arrays
    if any(a.ndim != 1 for a in arrays) or not (len(v) == len(i) == len(t)):
        raise ValueError("Signal shape mismatch")
    if len(t) < window.min_observations or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Missing or nonfinite signal")
    if (np.diff(t) <= 0).any():
        raise ValueError("Time must be strictly increasing within cycle")
    positive = i > 0
    boundaries = np.diff(np.r_[False, positive, False].astype(int))
    for start, stop in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
        for a in range(start, stop - 1):
            if not (v[a] <= window.low_v < v[a + 1]):
                continue
            crossings = np.flatnonzero((v[a + 1:stop - 1] < window.high_v)
                                      & (v[a + 2:stop] >= window.high_v))
            if not len(crossings):
                continue
            b = a + 1 + int(crossings[0])
            if b + 2 - a < window.min_observations:
                continue
            lo = t[a] + (t[a + 1] - t[a]) * (window.low_v - v[a]) / (v[a + 1] - v[a])
            hi = t[b] + (t[b + 1] - t[b]) * (window.high_v - v[b]) / (v[b + 1] - v[b])
            grid = np.linspace(lo, hi, window.points)
            x = np.stack([np.interp(grid, t[a:b + 2], v[a:b + 2]),
                          np.interp(grid, t[a:b + 2], i[a:b + 2]), grid - lo])
            return x.astype("float32"), {"duration_s": float(hi-lo),
                "sample_start": int(a), "sample_stop_exclusive": int(b+2),
                "cycle_number": float(cycle["cycle_number"])}
    raise ValueError("No complete contiguous charge window")


def support_reference(capacities, stable_reference_indices=None):
    """Use only supplied early support capacities, never query capacities.

    Explicit stable indices must be established by a dataset-specific audit.
    Without that evidence use the plan's first reliable-capacity fallback and
    record it, rather than asserting that the first three points are stable.
    """
    q = np.asarray(capacities, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or (q <= 0).any():
        raise ValueError("Reliable positive support capacities required")
    if stable_reference_indices is None:
        return float(q[0]), "first_reliable_support_capacity_fallback"
    idx = np.asarray(stable_reference_indices)
    if (idx.shape != (3,) or not np.issubdtype(idx.dtype, np.integer)
            or len(set(idx.tolist())) != 3 or (idx < 0).any() or (idx >= len(q)).any()):
        raise ValueError("Three distinct audited support indices required")
    return float(q[idx].mean()), "three_audited_stable_support_capacities"
