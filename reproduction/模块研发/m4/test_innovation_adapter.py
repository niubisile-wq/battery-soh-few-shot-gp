from types import SimpleNamespace
import numpy as np
import joblib
from test_conditional_mixed import setup
from innovation_adapter import InnovationAdapter


def test_innovation_identity_and_preserved_parent_difference(tmp_path):
    model,c=setup();p=model.predict(c);other=p+np.linspace(-.05,.03,len(p))
    for mode in ('cv','uniform'):
        m=InnovationAdapter(model,mode);delta=m.predict(c)
        np.testing.assert_allclose(m.apply(c,p,1),m.adapter.predict(c),atol=1e-12)
        np.testing.assert_array_equal(m.apply(c,p,0),p)
        np.testing.assert_allclose(m.apply(c,p,.5)-m.apply(c,other,.5),p-other,atol=1e-12)
        yy=c.y.copy();yy[10:]=np.nan
        np.testing.assert_allclose(delta,m.predict(SimpleNamespace(**dict(c.__dict__,y=yy))),atol=1e-12)
        short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
        np.testing.assert_allclose(delta[:3],m.predict(short),atol=1e-10)
        path=tmp_path/(mode+'.joblib');joblib.dump(m,path)
        np.testing.assert_array_equal(delta,joblib.load(path).predict(c))
