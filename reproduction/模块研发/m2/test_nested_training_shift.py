import numpy as np
import diagnose_nested_training_shift as ds


def test_identical_predictions_have_no_shift():
    r=ds.compare(np.array([1.,.8]),np.array([.9,.9]),np.array([.9,.9]))
    assert r['residual_shift_mae_pp']==0 and r['opposite_sign_fraction']==0


def test_changed_error_sign_and_shift_are_measured():
    r=ds.compare(np.array([1.,1.]),np.array([.8,.8]),np.array([1.1,1.1]))
    np.testing.assert_allclose(r['residual_shift_mae_pp'],30.)
    assert r['opposite_sign_fraction']==1 and r['shift_exceeds_two_error_fraction']==1
