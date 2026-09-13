from types import SimpleNamespace
import numpy as np
from scipy.linalg import block_diag,helmert
import joblib
from difference_branch import DifferenceBranch
from mixed_effects_branch import MixedEffectsBranch
from conditional_mixed import ConditionalMixed
from test_correction import fixture


def setup():
    cc,_=fixture();cells=[SimpleNamespace(**c.__dict__,reference_capacity=2.) for c in cc.values()]
    allowed={c.id:list(range(13)) for c in cells};masked=[]
    for c in cells:
        yy=c.y.copy();yy[13:]=np.nan
        masked.append(SimpleNamespace(**dict(c.__dict__,y=yy)))
    ref=DifferenceBranch('difference',optimize=False).fit(masked,allowed)
    model=MixedEffectsBranch.from_encoder(ref,masked,allowed,optimize=False)
    return model,cells[0]


def test_mask_prefix_serialization(tmp_path):
    model,c=setup()
    for private in (False,True):
        m=ConditionalMixed(model,private);p=m.predict(c);yy=c.y.copy();yy[10:]=999
        np.testing.assert_allclose(p,m.predict(SimpleNamespace(**dict(c.__dict__,y=yy))),atol=1e-12)
        short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
        np.testing.assert_allclose(p[:3],m.predict(short),atol=1e-10)
        path=tmp_path/(str(private)+'.joblib');joblib.dump(m,path)
        np.testing.assert_array_equal(p,joblib.load(path).predict(c))


def test_joint_dense_conditioning():
    model,c=setup();gp=model.gp
    for private in (False,True):
        m=ConditionalMixed(model,private);z=m.latent(c);zs=z[:10];hs=helmert(10,full=False)
        x=gp.X_train_;h=gp.H_;rho=gp.rho_ if private else 0.
        source_cov=h@(gp.kernel_(x)+gp.rho_*gp.slope_)@h.T
        cross=h@gp.kernel_(x,zs)@hs.T
        support_cov=hs@(gp.kernel_(zs)+rho*(zs@zs.T)/z.shape[1])@hs.T
        joint=np.block([[source_cov,cross],[cross.T,support_cov]])
        target=np.r_[gp.y_contrast_,hs@(c.y[:10]/gp.y_scale_)]
        weights=np.linalg.solve(joint,target)
        query_cross=np.c_[gp.kernel_(z,x)@h.T,(gp.kernel_(z,zs)+rho*(z@zs.T)/z.shape[1])@hs.T]
        pred=query_cross@weights*gp.y_scale_
        expected=pred[10:]+np.mean(c.y[:10]-pred[:10])
        np.testing.assert_allclose(m.predict(c),expected,atol=1e-10,rtol=1e-8)
