from types import SimpleNamespace
import numpy as np
from difference_branch import DifferenceBranch
from mixed_effects_branch import MixedEffectsBranch
from test_correction import fixture


def test_reused_encoder_budget_and_inference_boundary():
    cc,_=fixture();cells=[SimpleNamespace(**c.__dict__,reference_capacity=2.) for c in cc.values()]
    allowed={c.id:list(range(13)) for c in cells};masked=[]
    for c in cells:
        y=c.y.copy();y[13:]=np.nan
        masked.append(SimpleNamespace(**dict(c.__dict__,y=y)))
    ref=DifferenceBranch('difference',optimize=False).fit(masked,allowed)
    old=ref.predict(cells[0]).copy()
    zero=MixedEffectsBranch.from_encoder(ref,masked,allowed,learn_rho=False,optimize=False)
    np.testing.assert_allclose(old,zero.predict(cells[0]),atol=1e-12)
    new=MixedEffectsBranch.from_encoder(ref,masked,allowed,optimize=False)
    c=cells[0];p=new.predict(c);y=c.y.copy();y[10:]=999
    np.testing.assert_allclose(p,new.predict(SimpleNamespace(**dict(c.__dict__,y=y))),atol=1e-12)
    short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
    np.testing.assert_allclose(p[:3],new.predict(short),atol=1e-10)
    np.testing.assert_array_equal(old,ref.predict(c))
    np.testing.assert_array_equal(ref.gp.X_train_,new.gp.X_train_)
