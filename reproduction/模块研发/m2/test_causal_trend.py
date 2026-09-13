import numpy as np
import pytest
from causal_trend import filter_prediction


def test_zero_time_scale_is_exact_parent():
    p=np.array([.9,.85,.87])
    np.testing.assert_array_equal(filter_prediction(p,[11,12,13],1.,10,0,.2,.05),p)


def test_irregular_cycle_decay_matches_closed_form():
    t=np.array([11.,15.,30.]);p=np.full(3,.8)
    expected=.8+.2*np.exp(-(t-10)/20.)
    np.testing.assert_allclose(filter_prediction(p,t,1.,10,20.,0.,0.),expected,atol=1e-12)


def test_filter_does_not_change_past_when_future_changes():
    p=np.array([.98,.97,.95,.96,.91]);t=np.arange(11.,16.)
    full=filter_prediction(p,t,1.,10,5.,.2,.05)
    np.testing.assert_allclose(filter_prediction(p[:3],t[:3],1.,10,5.,.2,.05),full[:3])
    p[3:]=123.
    np.testing.assert_allclose(filter_prediction(p,t,1.,10,5.,.2,.05)[:3],full[:3])


def test_reversed_cycles_are_rejected():
    with pytest.raises(ValueError): filter_prediction([.9,.8],[11.,10.],1.,10,5.,.2,.05)
