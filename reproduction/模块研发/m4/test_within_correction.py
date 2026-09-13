from dataclasses import replace
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from within_correction import WithinCorrector, within_projection
from test_correction import fixture


def test_projection_and_kernel():
    labels=np.array(['a','a','b','b']);w=np.array([.1,.2,.3,.4])
    h=within_projection(labels,w)
    np.testing.assert_allclose(h@h,h,atol=1e-12)
    np.testing.assert_allclose(h@np.array([2,2,-3,-3]),0,atol=1e-12)
    z=np.arange(8).reshape(4,2);k=h@rbf_kernel(z)@h.T
    assert np.linalg.eigvalsh(k).min()>-1e-10


def test_labels_offsets_reference_prefix():
    cells,ee=fixture()
    ee=[dict(e,y=e['y']+np.linspace(-.02,.02,len(e['y']))) for e in ee]
    shifted=[dict(e,y=e['y']+(.2 if e['domain']=='0' else -.1)) for e in ee]
    for learner in ('ridge','rbf'):
        m=WithinCorrector('signal',learner,10).fit(ee,cells,'B123')
        other=WithinCorrector('signal',learner,10).fit(shifted,cells,'B123')
        c=cells['0'];p=ee[0]['predictions']['B123'];r=m.predict(c,p)
        np.testing.assert_allclose(r,other.predict(c,p),atol=1e-10)
        yy=c.y.copy();yy[10:]=999
        np.testing.assert_allclose(r,m.predict(replace(c,y=yy),p),atol=1e-10)
        short=replace(c,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13])
        np.testing.assert_allclose(r[:3],m.predict(short,p[:3]),atol=1e-10)
        # All curves identical to reference, parent equals reference SOH.
        x=np.broadcast_to(c.x[0],c.x.shape).copy();ref=replace(c,x=x)
        np.testing.assert_allclose(m.predict(ref,np.full(len(p),c.y[:10].mean())),0,atol=1e-10)
