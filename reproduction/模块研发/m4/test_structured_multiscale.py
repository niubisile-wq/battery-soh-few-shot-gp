from types import SimpleNamespace
import joblib
import numpy as np
from structured_multiscale import StructuredMultiscaleGP,column_groups
from conditional_geometry import ConditionalGeometry
from test_correction import fixture


def test_partition():
    shape,energy=column_groups()
    assert len(shape)==96 and len(energy)==72
    assert set(shape).isdisjoint(energy) and set(shape)|set(energy)==set(range(168))


def test_source_budget_prediction_and_serialization(tmp_path):
    cc,_=fixture();cells=[SimpleNamespace(**c.__dict__,reference_capacity=2.) for c in cc.values()]
    allowed={c.id:list(range(13)) for c in cells}
    visible=[]
    for c in cells:
        yy=c.y.copy();yy[13:]=np.nan;visible.append(SimpleNamespace(**dict(c.__dict__,y=yy)))
    for encoder,kernel in [('pca8','joint'),('pca8','additive'),('pls4','joint'),('pls4','additive')]:
        gp=StructuredMultiscaleGP(encoder,kernel).fit(visible,allowed)
        assert len(gp.gp.X_train_)==52 and set(gp.source_keys)=={(c.id,j) for c in cells for j in allowed[c.id]}
        for mode in ('source_bias','posterior'):
            model=ConditionalGeometry(gp,mode);c=cells[0];pred=model.predict(c)
            yy=c.y.copy();yy[10:]=999
            np.testing.assert_allclose(pred,model.predict(SimpleNamespace(**dict(c.__dict__,y=yy))),atol=1e-9)
            short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
            np.testing.assert_allclose(pred[:3],model.predict(short),atol=1e-9)
            path=tmp_path/(encoder+kernel+mode+'.joblib');joblib.dump(model,path)
            np.testing.assert_array_equal(pred,joblib.load(path).predict(c))
