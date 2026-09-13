"""XJTU measured-capacity eligibility, based on protocol metadata and signals.

No capacity-magnitude filtering to remove poor predictions. Mixed aging/RPT
records use explicit RPT only. Full-discharge-only protocols can use cycling
records if they reach the recorded cutoff under load. Never interpolate labels.
"""
import numpy as np


def xjtu_capacity(cycle, cell, has_rpt):
    if has_rpt and cycle.get('attribute') != 'RPT':
        raise ValueError('Aging discharge is not a reference capacity measurement')
    v=np.asarray(cycle['voltage_in_V'],dtype=float)
    i=np.asarray(cycle['current_in_A'],dtype=float)
    q=np.asarray(cycle['discharge_capacity_in_Ah'],dtype=float)
    if v.ndim!=1 or i.shape!=v.shape or q.shape!=v.shape:
        raise ValueError('Capacity and signal shape mismatch')
    if not all(np.isfinite(a).all() for a in (v,i,q)):
        raise ValueError('Nonfinite capacity measurement')
    cutoff=float(cell['min_voltage_limit_in_V'])
    discharge=i < -.05 * float(cell['nominal_capacity_in_Ah'])
    if not discharge.any() or v[discharge].min() > cutoff+.02:
        raise ValueError('Discharge did not reach reference cutoff under load')
    capacity=float(q[discharge].max())
    if capacity<=0:
        raise ValueError('Nonpositive measured capacity')
    return capacity
