from dataclasses import replace
import numpy as np
import pytest
import nested_signal_probe as ns
from test_m1 import toy


def test_signal_features_ignore_labels_future_and_preserve_prefix():
    c=toy(0);q=np.array([10,11,12])
    original=ns.signal_views(c,q)
    x=c.x.copy();x[13:]=999
    changed=ns.signal_views(replace(c,x=x,y=np.full_like(c.y,np.nan)),q)
    short=ns.signal_views(replace(c,x=c.x[:13],y=c.y[:13]),q[:2])
    for key in original:
        np.testing.assert_array_equal(original[key],changed[key])
        np.testing.assert_array_equal(original[key][:2],short[key])


def test_weights_equal_domains_then_cells_then_queries():
    records=[dict(domain='a',y=np.zeros(2)),dict(domain='a',y=np.zeros(3)),dict(domain='b',y=np.zeros(4))]
    w=ns.balanced_weights(records)
    np.testing.assert_allclose([w[:2].sum(),w[2:5].sum(),w[5:].sum()],[.25,.25,.5])


@pytest.mark.parametrize('method',ns.METHODS)
def test_mapping_never_uses_validation_labels_or_predictions(method):
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(3)]
    views={c.id:ns.signal_views(c,np.array([10,11,12])) for c in cells}
    records=[dict(cell_id=c.id,domain=c.domain,y=np.array([.9,.8,.7]),
                  base=np.array([.95,.85,.75]),physical=np.array([.91,.81,.71])) for c in cells]
    a,_=ns.learn_predict(records[:2],records[2:],method,views)
    changed=[dict(records[2],y=np.full(3,np.nan),base=np.full(3,np.nan),physical=np.full(3,np.nan))]
    b,_=ns.learn_predict(records[:2],changed,method,views)
    np.testing.assert_allclose(a,b)
    assert np.isfinite(a).all()
