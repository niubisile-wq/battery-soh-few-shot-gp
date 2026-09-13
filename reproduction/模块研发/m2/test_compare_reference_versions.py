import pytest
from compare_reference_versions import paired_draws


def test_draw_pairing_is_by_id_not_array_position():
    a=dict(resamples=[dict(fold='f',group='g',seed=1,source_ids=['a','b'],counts=[1,2])])
    b=dict(resamples=[dict(fold='f',group='g',seed=1,source_ids=['b','a'],counts=[2,1])])
    assert paired_draws(a,b)==1
    b['resamples'][0]['counts']=[1,2]
    with pytest.raises(AssertionError):paired_draws(a,b)
