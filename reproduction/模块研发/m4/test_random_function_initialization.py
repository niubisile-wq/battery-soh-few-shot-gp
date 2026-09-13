from types import SimpleNamespace
import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from random_function_fit_pilot import initialized_reference


@pytest.mark.parametrize('base', [RBF(1.), Matern(1., nu=1.5)])
def test_only_prior_length_changes_without_mutating_reference(base):
    kernel = ConstantKernel(1.) * base + WhiteKernel(.01)
    ref = SimpleNamespace(gp=SimpleNamespace(kernel=kernel), source_keys=[('a', 0)])
    original = kernel.theta.copy()
    for scale in (1., .5, .8, 1.25, 2.):
        out, info = initialized_reference(ref, scale)
        np.testing.assert_array_equal(kernel.theta, original)
        np.testing.assert_array_equal(out.gp.kernel.theta[[0, 2]], original[[0, 2]])
        assert out.gp.kernel.theta[1] == pytest.approx(original[1] + np.log(scale))
        assert out.source_keys == ref.source_keys and out is not ref
        assert not info['clipped']
    with pytest.raises(ValueError): initialized_reference(ref, 0)
