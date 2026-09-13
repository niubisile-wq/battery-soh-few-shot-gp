from dataclasses import replace
import numpy as np
from centered_correction import CenteredCorrector
from test_correction import fixture


def test_domain_offset_and_shift_invariance():
    cells, ee = fixture()
    original = [e['y'].copy() for e in ee]
    shifted = [dict(e, y=e['y'] + (.3 if e['domain']=='0' else -.2)) for e in ee]
    for learner in ('constant','ridge','rbf'):
        a=CenteredCorrector('signal',learner,10).fit(ee,cells,'B123')
        b=CenteredCorrector('signal',learner,10).fit(shifted,cells,'B123')
        assert abs(a.mean)<1e-10
        np.testing.assert_allclose([a.domain_offsets['0'],a.domain_offsets['1']],[-.03,-.01])
        for e in ee:
            c=cells[e['cell_id']];p=e['predictions']['B123']
            np.testing.assert_allclose(a.predict(c,p),b.predict(c,p),atol=1e-10)
    for e,y in zip(ee,original):np.testing.assert_array_equal(e['y'],y)


def test_mask_and_prefix():
    cells,ee=fixture()
    # Add within-domain structure so a zero residual alone cannot satisfy this test.
    ee=[dict(e,y=e['y']+np.linspace(-.02,.02,len(e['y']))) for e in ee]
    for learner in ('ridge','rbf'):
        m=CenteredCorrector('signal',learner,10).fit(ee,cells,'B123')
        c=cells['0'];p=ee[0]['predictions']['B123'];r=m.predict(c,p)
        assert np.max(abs(r))>1e-6 and np.max(abs(r))<=.05
        yy=c.y.copy();yy[10:]=999
        np.testing.assert_allclose(r,m.predict(replace(c,y=yy),p))
        short=replace(c,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13])
        np.testing.assert_allclose(r[:3],m.predict(short,p[:3]))
