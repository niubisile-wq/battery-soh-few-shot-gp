import unittest
from dataclasses import replace
import numpy as np
import test_geometry
from signal_trajectory import correction
from screen import sa


class SignalTrajectoryTests(unittest.TestCase):
    def test_boundaries(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        p=c.y[10:]+.003*np.sin(np.arange(18))
        for kind in ['signal_5_0.1','signal_15_1','signal_31_0.1']:
            r=correction(c,p,kind)
            changed=c.y.copy();changed[10:]=123
            np.testing.assert_array_equal(r,correction(replace(c,y=changed),p,kind))
            for n in [11,17,24]:
                np.testing.assert_allclose(r[:n-10],correction(sa.prefix(c,n),p[:n-10],kind),rtol=0,atol=1e-8)
            altered=c.x.copy();altered[20:,1]*=3
            np.testing.assert_array_equal(r[:10],correction(replace(c,x=altered),p,kind)[:10])


if __name__=='__main__':unittest.main()
