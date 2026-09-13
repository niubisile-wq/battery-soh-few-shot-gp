import unittest
from dataclasses import replace
import numpy as np
import test_geometry
from support_clock import reference_trend,clock_features,ordinary_trend,SupportClockGP
from conditional_geometry import ConditionalGeometry
from screen import sa
from threadpoolctl import threadpool_limits


class ClockTests(unittest.TestCase):
    def test_linear_and_boundaries(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        np.testing.assert_allclose(ordinary_trend(c),c.y[10:],atol=1e-10)
        for view in ['clock_only','clock_joint']:
            f=clock_features(c,view);y=c.y.copy();y[10:]=123
            np.testing.assert_array_equal(f,clock_features(replace(c,y=y),view))
            np.testing.assert_array_equal(f[:13],clock_features(sa.prefix(c,13),view))
        changed=c.y.copy();changed[:10]+=.01*np.arange(10)
        self.assertNotEqual(reference_trend(c)[2],reference_trend(replace(c,y=changed))[2])

    def test_model_prefix(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        allowed={c.id:list(range(10))+[12,15,20,25]}
        with threadpool_limits(limits=1):
            parent=SupportClockGP('clock_joint','rbf').fit([sa.source_view(c,allowed[c.id])],allowed)
            for mode in ['source_bias','posterior']:
                m=ConditionalGeometry(parent,mode);p=m.predict(sa.inference_view(c))
                np.testing.assert_allclose(p[:3],m.predict(sa.prefix(c,13)),rtol=0,atol=1e-8)


if __name__=='__main__':unittest.main()
