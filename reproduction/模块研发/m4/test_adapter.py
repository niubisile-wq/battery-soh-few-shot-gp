from dataclasses import replace
import joblib
import numpy as np
import pytest
from adapter import ResidualM4Adapter
from correction import ResidualCorrector
from test_correction import fixture


class Parent:
    def predict(self,c,mode=None):
        assert np.isnan(c.y[10:]).all()
        return np.mean(c.y[:10])+.01*np.asarray(c.x[10:,0,0])


def test_end_to_end_mask_prefix_and_serialization(tmp_path):
    cells,ee=fixture();c=cells['0'];parent=Parent()
    corrector=ResidualCorrector('signal','rbf',10).fit(ee,cells,'B123')
    model=ResidualM4Adapter(parent,corrector,.25,'source_only');p=model.predict(c)
    y=c.y.copy();y[10:]=123
    np.testing.assert_allclose(p,model.predict(replace(c,y=y)),atol=1e-10)
    short=replace(c,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13])
    np.testing.assert_allclose(p[:3],model.predict(short),atol=1e-10)
    path=tmp_path/'adapter.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(p,joblib.load(path).predict(c))


def test_gain_validation_and_identity():
    cells,_=fixture();c=cells['0'];parent=Parent()
    for gain in [-1,2,float('nan')]:
        with pytest.raises(ValueError):ResidualM4Adapter(parent,None,gain)
    model=ResidualM4Adapter(parent,None,0)
    np.testing.assert_allclose(model.predict(c),np.mean(c.y[:10])+.01*c.x[10:,0,0])
