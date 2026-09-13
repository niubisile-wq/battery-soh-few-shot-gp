from types import SimpleNamespace
import numpy as np
import joblib
from test_correction import fixture
from difference_branch import DifferenceBranch


def test_budget_prefix_labels_serialization_and_matched_encoder(tmp_path):
    cc,_=fixture();cells=[SimpleNamespace(**c.__dict__,reference_capacity=2.) for c in cc.values()]
    allowed={c.id:list(range(13)) for c in cells};masked=[];models=[]
    for c in cells:
        y=c.y.copy();y[13:]=np.nan
        masked.append(SimpleNamespace(**dict(c.__dict__,y=y)))
    for target in ('direct','difference'):
        m=DifferenceBranch(target,optimize=False).fit(masked,allowed);models.append(m)
        assert len(m.source_keys)==52
        c=cells[0];p=m.predict(c);y=c.y.copy();y[10:]=999
        np.testing.assert_allclose(p,m.predict(SimpleNamespace(**dict(c.__dict__,y=y))),atol=1e-12)
        short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
        np.testing.assert_allclose(p[:3],m.predict(short),atol=1e-10)
        path=tmp_path/(target+'.joblib');joblib.dump(m,path)
        np.testing.assert_array_equal(p,joblib.load(path).predict(c))
    np.testing.assert_array_equal(models[0].gp.X_train_,models[1].gp.X_train_)
    np.testing.assert_allclose(models[0].scale,models[1].gp.y_scale_)
