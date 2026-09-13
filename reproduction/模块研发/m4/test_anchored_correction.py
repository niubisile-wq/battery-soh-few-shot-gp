from dataclasses import replace
import numpy as np
from anchored_correction import AnchoredCorrector,difference_kernel
from test_correction import fixture


def test_kernel_positive_semidefinite_and_zero():
    rng=np.random.default_rng(5);z=rng.normal(size=(12,4));z0=rng.normal(size=(12,4))
    k=difference_kernel(z,z0,z,z0,.25)
    np.testing.assert_allclose(k,k.T,atol=1e-12)
    assert np.linalg.eigvalsh(k).min()>-1e-10
    np.testing.assert_allclose(difference_kernel(z0,z0,z,z0,.25),0,atol=1e-12)


def test_reference_zero_label_and_prefix():
    cells,ee=fixture();c=cells['0'];p=ee[0]['predictions']['B123']
    for view in ['state','signal']:
        for learner in ['ridge','rbf']:
            model=AnchoredCorrector(view,learner,10).fit(ee,cells,'B123');r=model.predict(c,p)
            y=c.y.copy();y[10:]=456
            np.testing.assert_allclose(r,model.predict(replace(c,y=y),p),atol=1e-10)
            short=replace(c,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13])
            np.testing.assert_allclose(r[:3],model.predict(short,p[:3]),atol=1e-10)
            # Identical first10/current waveforms and anchor prediction define the reference state.
            x=np.broadcast_to(c.x[:1],c.x.shape).copy();reference=replace(c,x=x)
            np.testing.assert_allclose(model.predict(reference,np.full(len(p),c.y[:10].mean())),0,atol=1e-10)
