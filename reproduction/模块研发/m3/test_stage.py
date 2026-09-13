import unittest
import numpy as np
from stage_insertion import combine


class StageTests(unittest.TestCase):
    def test_affine_insertion(self):
        b=np.array([.8,.7]);b1=b-.01;phy=np.array([.82,.73]);a=.4;res=.003
        p={'B':b,'B1':b1,'B2':a*b+(1-a)*phy+res,'B12':a*b1+(1-a)*phy+res}
        branch=np.array([.75,.68]);w=.5
        result=combine(p,branch,w,{'B2':a,'B12':a})
        np.testing.assert_allclose(result['B123'],a*((1-w)*b1+w*branch)+(1-a)*phy+res)
        for g in p:np.testing.assert_array_equal(combine(p,branch,0,{'B2':a,'B12':a})[g+'3'],p[g])


if __name__=='__main__':unittest.main()
