import unittest
import numpy as np
import test_geometry
from screen import sa
from support_clock import SupportClockGP
from conditional_geometry import ConditionalGeometry
from coverage_gate import coverage,trust
from threadpoolctl import threadpool_limits


class CoverageTests(unittest.TestCase):
    def test_limits_and_prefix(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        allowed={c.id:list(range(10))+[12,15,20,25]}
        with threadpool_limits(limits=1):
            parent=SupportClockGP('clock_joint','rbf').fit([sa.source_view(c,allowed[c.id])],allowed)
            m=ConditionalGeometry(parent,'source_bias');q,r=coverage(m,c);qs,rs=coverage(m,sa.prefix(c,13))
            np.testing.assert_allclose(q[:3],qs,rtol=0,atol=1e-10);self.assertAlmostEqual(r,rs)
            self.assertTrue(np.all((q>=0)&(q<=1)));self.assertTrue(0<=r<=1)
            self.assertTrue(np.all(trust(q,r,'product',2)<=trust(q,r,'query',1)+1e-12))


if __name__=='__main__':unittest.main()
