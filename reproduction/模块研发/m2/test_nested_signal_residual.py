import numpy as np
import nested_signal_residual as sr


def test_anchored_kernel_is_psd_and_zero_at_reference():
    x=np.random.default_rng(0).normal(size=(12,4));k=sr.kernel(x,x,True)
    assert np.linalg.eigvalsh(k).min()>-1e-12
    np.testing.assert_allclose(sr.kernel(np.zeros((1,4)),x,True),0,atol=1e-15)


def test_signed_residual_fit_learns_direction_and_anchor():
    x=np.tile(np.linspace(.1,1,20)[:,None],(1,21))
    records=[dict(cell_id='a',domain='a',base=np.ones(20),y=1-.1*x[:,0])]
    head=sr.fit(records,{'a':dict(delta=x)},True)
    assert sr.correction(x,head).mean()<-.02
    np.testing.assert_allclose(sr.correction(np.zeros((1,21)),head),0,atol=1e-12)
