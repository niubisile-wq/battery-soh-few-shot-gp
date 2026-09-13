import unittest
from dataclasses import replace
import numpy as np
import test_geometry
from screen import sa
from reliability import ReliabilityGate


class ReliabilityTests(unittest.TestCase):
    def test_fit_and_boundaries(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        idx=np.array([12,15,20,25]);allowed={c.id:list(range(10))+idx.tolist()}
        e=dict(cell_id=c.id,domain=c.domain,base=np.full(4,.99),y=c.y[idx])
        for view in ['state','signal']:
            model=ReliabilityGate(view,10).fit([e],{c.id:c.y[idx]+.01},[sa.source_view(c,allowed[c.id])],allowed)
            p=np.full(18,.99);b=c.y[10:]+.01;w=model.predict(sa.inference_view(c),p,b)
            y=c.y.copy();y[10:]=123
            np.testing.assert_array_equal(w,model.predict(replace(c,y=y),p,b))
            np.testing.assert_allclose(w[:3],model.predict(sa.prefix(c,13),p[:3],b[:3]),rtol=0,atol=1e-10)
            self.assertTrue(np.all((w>=0)&(w<=.75)))
            self.assertEqual(set(model.source_keys),{(c.id,int(j)) for j in idx})


if __name__=='__main__':unittest.main()
