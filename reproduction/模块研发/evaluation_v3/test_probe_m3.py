import unittest
from types import SimpleNamespace
import numpy as np
from probe_m3 import correction


class ProbeTests(unittest.TestCase):
    def test_causal_and_zero(self):
        x=np.ones((15,3,128));x[:,2]=np.linspace(0,100,128)
        c=SimpleNamespace(x=x,y=np.r_[np.ones(10),np.zeros(5)])
        p=np.linspace(.99,.8,5)
        for direction in ['degradation_gain','charge_anchor']:
            np.testing.assert_array_equal(correction(c,p,direction,0),p)
            a=correction(c,p,direction,.5)
            c.y[10:]=123
            np.testing.assert_array_equal(correction(c,p,direction,.5),a)
            short=SimpleNamespace(x=x[:12],y=c.y[:12])
            np.testing.assert_array_equal(correction(short,p[:2],direction,.5),a[:2])

    def test_recovery_not_suppressed(self):
        c=SimpleNamespace(x=np.zeros((12,3,128)),y=np.ones(12))
        np.testing.assert_array_equal(correction(c,np.array([1.01,1.02]),'degradation_gain',.5),[1.01,1.02])


if __name__=='__main__':unittest.main()
