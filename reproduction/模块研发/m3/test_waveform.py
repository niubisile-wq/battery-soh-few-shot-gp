import unittest
from dataclasses import replace
import numpy as np
import test_geometry
from screen import sa
from waveform import waveform_features,WaveformGP
from conditional_geometry import ConditionalGeometry
from threadpoolctl import threadpool_limits


class WaveformTests(unittest.TestCase):
    def test_reference_pair_and_no_age(self):
        f=test_geometry.GeometryTests();f.setUp();c=f.cell
        for view in ['wave_absolute','wave_pair']:
            z=waveform_features(c,view)
            np.testing.assert_array_equal(z[:13],waveform_features(sa.prefix(c,13),view))
            np.testing.assert_array_equal(z,waveform_features(replace(c,y=np.full(len(c.y),np.nan),cycle=c.cycle*100),view))
        z=waveform_features(c,'wave_pair');np.testing.assert_allclose(z[:10,384:].mean(0),0,atol=1e-12)

    def test_projection_and_prediction_prefix(self):
        f=test_geometry.GeometryTests();f.setUp();c=f.cell;allowed={c.id:list(range(10))+[12,15,20,25]}
        with threadpool_limits(limits=1):
            for encoder in ['pca8','pls4']:
                p=WaveformGP('wave_pair_'+encoder,'rbf').fit([sa.source_view(c,allowed[c.id])],allowed)
                for mode in ['source_bias','posterior']:
                    m=ConditionalGeometry(p,mode);y=m.predict(sa.inference_view(c))
                    np.testing.assert_allclose(y[:3],m.predict(sa.prefix(c,13)),rtol=0,atol=1e-8)


if __name__=='__main__':unittest.main()
