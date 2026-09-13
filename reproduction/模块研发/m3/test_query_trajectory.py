import unittest
from dataclasses import replace
import numpy as np
import test_geometry
from query_trajectory import correction
from trajectory_screen import P,sa


class QueryTests(unittest.TestCase):
    def test_boundaries_and_no_support_splice(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        p=c.y[10:]+.003*np.sin(np.arange(18))
        for kind in P['candidates']:
            r=correction(c,p,kind)
            np.testing.assert_array_equal(r,correction(replace(c,y=np.full(len(c.y),123)),p,kind))
            for n in [11,17,24]:
                np.testing.assert_allclose(r[:n-10],correction(sa.prefix(c,n),p[:n-10],kind),rtol=0,atol=1e-8)
            np.testing.assert_allclose(correction(c,np.full(len(p),.7),kind),0,rtol=0,atol=1e-10)


if __name__=='__main__':unittest.main()
