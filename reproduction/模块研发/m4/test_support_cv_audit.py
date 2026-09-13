"""Integration checks for the independent auditor and saved deployment adapter."""
from types import SimpleNamespace
import joblib
import numpy as np
from test_conditional_mixed import setup
from support_cv_mixed import SupportCVMixed
from audit_support_cv import independent_weight


def test_independent_weight_and_future_input_invariance(tmp_path):
    parent,c=setup();model=SupportCVMixed(parent)
    actual=model.weights(c);independent=independent_weight(parent,c)
    for key,value in actual.items():
        np.testing.assert_allclose(value,independent[key],rtol=0,atol=1e-8)
    x=c.x.copy();x[10:]=np.nan
    y=c.y.copy();y[10:]=np.nan
    changed=SimpleNamespace(**dict(c.__dict__,x=x,y=y,id='unseen-name',domain='unseen-domain'))
    assert model.weights(changed)==actual
    for key,value in independent.items():
        np.testing.assert_allclose(value,independent_weight(parent,changed)[key],rtol=0,atol=1e-12)
    path=tmp_path/'support_cv.joblib';joblib.dump(model,path)
    restored=joblib.load(path)
    assert restored.weights(c)==actual
    np.testing.assert_array_equal(model.predict(c),restored.predict(c))
