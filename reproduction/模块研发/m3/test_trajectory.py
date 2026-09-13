import unittest
from dataclasses import replace
import numpy as np
import test_geometry
from trajectory import correction
from trajectory_screen import P,sa


class TrajectoryTests(unittest.TestCase):
    def test_boundaries(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        p=c.y[10:]+.003*np.sin(np.arange(18))
        for kind in P['candidates']:
            r=correction(c,p,kind)
            changed=c.y.copy();changed[10:]=123
            np.testing.assert_array_equal(r,correction(replace(c,y=changed),p,kind))
            for n in [11,17,24]:
                np.testing.assert_allclose(r[:n-10],correction(sa.prefix(c,n),p[:n-10],kind),rtol=0,atol=1e-10)
            self.assertTrue(np.max(abs(r))<=.05)

    def test_linear_no_lag_and_recovery(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        for direction in [-1,1]:
            y=.8+direction*.002*np.arange(len(c.y));c=replace(c,y=y)
            for kind in ['linear_5','linear_15','linear_31']:
                np.testing.assert_allclose(correction(c,y[10:],kind),0,rtol=0,atol=1e-10)


if __name__=='__main__':unittest.main()
