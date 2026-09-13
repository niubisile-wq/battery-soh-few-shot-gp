from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from support_gate import SupportGate
from test_correction import fixture


class Identity:
    def transform(self,x):return x


class Fake:
    def __init__(self):
        self.gp=SimpleNamespace(X_train_=np.array([[0.],[.1],[2.],[2.1]]))
        self.source_keys=[('a',0),('a',1),('b',0),('b',1)]
        self.scaler=Identity()
    def feature_matrix(self,c):return c.x[:,0,:1]


def test_scale_excludes_same_cell():
    gate=SupportGate(Fake())
    np.testing.assert_allclose(gate.scale2,(4+3.61)/2)
    cells,_=fixture();c=cells['0'];x=c.x.copy();x[:,0,0]=10
    far=gate.predict(replace(c,x=x));x[:,0,0]=0
    near=gate.predict(replace(c,x=x))
    assert np.all(near>far) and np.all(far>0) and np.all(near<=1)


def test_labels_prefix_and_source_only_scale():
    gate=SupportGate(Fake());cells,_=fixture();c=cells['0'];p=gate.predict(c)
    yy=c.y.copy();yy[10:]=999
    np.testing.assert_array_equal(p,gate.predict(replace(c,y=yy)))
    short=replace(c,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13])
    np.testing.assert_allclose(p[:3],gate.predict(short))
