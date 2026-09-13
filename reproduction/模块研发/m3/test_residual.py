from dataclasses import replace
import unittest
import numpy as np
import test_geometry
from residual import ResidualModel
from screen import sa
from threadpoolctl import threadpool_limits


class ResidualTests(unittest.TestCase):
    def test_masked_labels_and_prefix(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        allowed={c.id:list(range(10))+[12,15,20,25]};idx=np.array([12,15,20,25])
        e=dict(cell_id=c.id,y=c.y[idx],base=np.full(4,.99))
        with threadpool_limits(limits=1):
            for learner in ['ridge','gp']:
                model=ResidualModel('relative',learner).fit([e],[sa.source_view(c,allowed[c.id])],allowed)
                p=np.full(len(c.y)-10,.99);r=model.predict(sa.inference_view(c),p)
                alt=c.y.copy();alt[10:]=123
                np.testing.assert_allclose(r,model.predict(replace(c,y=alt),p),rtol=0,atol=1e-8)
                np.testing.assert_allclose(r[:3],model.predict(sa.prefix(c,13),p[:3]),rtol=0,atol=1e-8)
                self.assertEqual(set(model.source_keys),{(c.id,int(j)) for j in idx})


if __name__=='__main__':unittest.main()
