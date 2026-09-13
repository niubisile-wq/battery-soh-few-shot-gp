from types import SimpleNamespace
import joblib
import numpy as np
from test_conditional_mixed import setup
from test_correction import fixture
from difference_branch import DifferenceBranch
from hierarchical_branch import fit_branch


def test_budget_target_id_invariance_and_serialization(tmp_path):
    cc,_=fixture();cells=[SimpleNamespace(**c.__dict__,reference_capacity=2.) for c in cc.values()]
    allowed={c.id:list(range(13)) for c in cells};masked=[]
    for c in cells:
        y=c.y.copy();y[13:]=np.nan
        masked.append(SimpleNamespace(**dict(c.__dict__,y=y,domain='a' if c.id in ('0','1') else 'b')))
    ref=DifferenceBranch('difference',optimize=False).fit(masked,allowed)
    for hierarchical in (False,True):
        model=fit_branch(ref,masked,allowed,hierarchical,optimize=False);c=cells[0];p=model.predict(c)
        y=c.y.copy();y[10:]=999
        altered=SimpleNamespace(**dict(c.__dict__,y=y,id='new',domain='unseen'))
        np.testing.assert_allclose(p,model.predict(altered),atol=1e-12)
        short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
        np.testing.assert_allclose(p[:3],model.predict(short),atol=1e-10)
        assert len(model.source_keys)==52
        path=tmp_path/(str(hierarchical)+'.joblib');joblib.dump(model,path)
        np.testing.assert_array_equal(p,joblib.load(path).predict(c))
