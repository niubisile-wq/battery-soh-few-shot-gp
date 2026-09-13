import unittest
from dataclasses import replace
import numpy as np
from threadpoolctl import threadpool_limits
import test_geometry
from geometry import GeometryGP
from conditional_geometry import ConditionalGeometry
from screen import sa


class ConditionalTests(unittest.TestCase):
    def test_boundaries_and_original_control(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        allowed={c.id:list(range(10))+[12,15,20,25]}
        with threadpool_limits(limits=1):
            parent=GeometryGP('relative','rbf').fit([sa.source_view(c,allowed[c.id])],allowed)
            np.testing.assert_allclose(parent.predict(c),ConditionalGeometry(parent,'source_bias').predict(c),rtol=0,atol=1e-10)
            for mode in ['source_only','source_bias','posterior']:
                model=ConditionalGeometry(parent,mode);p=model.predict(sa.inference_view(c))
                changed=c.y.copy();changed[10:]=123
                np.testing.assert_allclose(p,model.predict(replace(c,y=changed)),rtol=0,atol=1e-8)
                np.testing.assert_allclose(p[:3],model.predict(sa.prefix(c,13)),rtol=0,atol=1e-8)


if __name__=='__main__':unittest.main()
