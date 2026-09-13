from types import SimpleNamespace
import numpy as np
from registered_charge import charge_features


def fixture():
    x=np.zeros((13,3,128));x[:,0]=np.linspace(3.5,4.1,128);x[:,1]=2.;x[:,2]=np.linspace(0,3600,128)
    return SimpleNamespace(x=x,reference_capacity=2.,y=np.ones(13))


def test_charge_and_prefix_and_label_independence():
    c=fixture();f=charge_features(c);assert f.shape==(13,198)
    np.testing.assert_allclose(f[:,63],1.)
    np.testing.assert_allclose(f[:,64:128],0,atol=1e-12)
    short=SimpleNamespace(x=c.x[:11],reference_capacity=2.,y=np.ones(11))
    np.testing.assert_array_equal(f[:11],charge_features(short))
    changed=SimpleNamespace(x=c.x,reference_capacity=2.,y=np.full(13,999))
    np.testing.assert_array_equal(f,charge_features(changed))


def test_fixed_grid_coverage_and_rate_invariance():
    c=fixture();base=charge_features(c);x=c.x.copy()
    x[10:,1]*=2;x[10:,2]/=2
    altered=charge_features(SimpleNamespace(x=x,reference_capacity=2.))
    np.testing.assert_allclose(base[:,:192],altered[:,:192])
    x=c.x.copy();x[10:,0]=np.linspace(3.6,4.0,128)
    f=charge_features(SimpleNamespace(x=x,reference_capacity=2.))
    assert np.all(f[10:,128]==0) and np.all(f[10:,191]==0)
    np.testing.assert_array_equal(f[:10],base[:10])
