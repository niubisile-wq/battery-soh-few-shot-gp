from dataclasses import replace
import numpy as np
from condition_correction import ConditionCorrector,parts,_RAW_CACHE
from geometry import features as geometry_features
from test_correction import fixture


def test_condition_fit_label_mask_and_prefix():
    cells,ee=fixture()
    for c in cells.values():
        c.x[:,0]=np.linspace(3.3,4.2,128)
        c.x[:,1]=abs(c.x[:,1])+.1
    hidden={cid:replace(c,y=np.r_[c.y[:10],np.full(len(c.y)-10,np.nan)]) for cid,c in cells.items()}
    for view in ['geometry_raw','geometry_residualized']:
        for learner in ['ridge','rbf']:
            model=ConditionCorrector(view,learner,10).fit(ee,hidden,'B123');c=cells['0'];p=ee[0]['predictions']['B123'];r=model.predict(c,p)
            np.testing.assert_allclose(r,model.predict(hidden['0'],p),atol=1e-10)
            short=replace(c,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13])
            np.testing.assert_allclose(r[:3],model.predict(short,p[:3]),atol=1e-10)
            assert np.isfinite(r).all() and abs(r).max()<=.05


def test_sparse_raw_extraction_and_cache_content_identity():
    cells,_=fixture();c=cells['0'];c.x[:,0]=np.linspace(3.3,4.2,128);c.x[:,1]=abs(c.x[:,1])+.1
    idx=np.array([10,14,19]);_RAW_CACHE.clear();g,context=parts(c,idx)
    np.testing.assert_allclose(g,geometry_features(c,'relative')[idx],atol=1e-12)
    before=g.copy();c.x[19,1]*=2
    changed,_=parts(c,idx)
    assert not np.allclose(before,changed)
    assert len(_RAW_CACHE)==2
