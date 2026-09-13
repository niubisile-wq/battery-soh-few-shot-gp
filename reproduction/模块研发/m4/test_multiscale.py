from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from multiscale import local_features, MultiscaleGP
from test_correction import fixture


def test_feature_shape_reference_prefix_and_labels():
    cells,_=fixture();c=cells['0']
    c=SimpleNamespace(**c.__dict__,reference_capacity=2.)
    f=local_features(c);assert f.shape==(20,168) and np.isfinite(f).all()
    np.testing.assert_allclose(f[:10,84:].mean(0),0,atol=1e-10)
    short=SimpleNamespace(**dict(c.__dict__,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13]))
    np.testing.assert_allclose(f[:13],local_features(short),atol=1e-10)
    altered=SimpleNamespace(**dict(c.__dict__,y=np.full(20,999.)))
    np.testing.assert_array_equal(f,local_features(altered))


def test_local_energy_detects_oscillation():
    x=np.zeros((12,3,128));x[:,0]=3.7;x[:,1]=2.;x[:,2]=np.arange(128)
    c=SimpleNamespace(x=x,reference_capacity=2.)
    original=local_features(c,False)
    x=x.copy();x[10:,0,::2]+=.01;x[10:,0,1::2]-=.01
    changed=local_features(SimpleNamespace(x=x,reference_capacity=2.),False)
    # Alternating perturbation leaves coarse signed Haar coefficients unchanged.
    np.testing.assert_allclose(original[10:,:16],changed[10:,:16],atol=1e-10)
    assert np.max(abs(original[10:,16:20]-changed[10:,16:20]))>.001


def test_source_budget_and_conditional_prediction(tmp_path):
    import joblib
    from conditional_geometry import ConditionalGeometry
    cells,_=fixture()
    cells=[SimpleNamespace(**c.__dict__,reference_capacity=2.) for c in cells.values()]
    allowed={c.id:list(range(13)) for c in cells}
    masked=[]
    for c in cells:
        yy=c.y.copy();yy[13:]=np.nan
        masked.append(SimpleNamespace(**dict(c.__dict__,y=yy)))
    for encoder in ('pca8','pls4'):
        gp=MultiscaleGP(encoder,'rbf').fit(masked,allowed)
        assert set(gp.source_keys)=={(c.id,j) for c in cells for j in allowed[c.id]}
        assert len(gp.gp.X_train_)==52
        model=ConditionalGeometry(gp,'source_bias');c=cells[0]
        expected=model.predict(c)
        yy=c.y.copy();yy[10:]=999
        np.testing.assert_allclose(expected,model.predict(SimpleNamespace(**dict(c.__dict__,y=yy))))
        path=tmp_path/(encoder+'.joblib');joblib.dump(model,path)
        np.testing.assert_array_equal(expected,joblib.load(path).predict(c))
