import unittest
import numpy as np
from measurement import repair_clock,events


class MeasurementTests(unittest.TestCase):
    def test_one_hour_clock_rollback(self):
        a=np.zeros((4,7));a[:,0]=np.arange(4);a[:,1]=[0,30,60,90];a[:,2]=[10000,10030,6460,6490]
        b,c=repair_clock(a)
        np.testing.assert_allclose(np.diff(b[:,2]),30)
        np.testing.assert_array_equal(a[:,4:],b[:,4:]);self.assertEqual(len(c),1)

    def test_unresolved_backwards_clock_rejected(self):
        a=np.zeros((3,7));a[:,0]=np.arange(3);a[:,1]=[0,30,60];a[:,2]=[1000,1030,900]
        with self.assertRaises(ValueError):repair_clock(a)

    def test_full_vs_partial_discharge(self):
        n=402;a=np.zeros((n,7));a[:,0]=np.arange(n);a[:,1]=np.arange(n)*10;a[:,2]=10000+a[:,1]
        a[:201,4]=.75;a[:201,5]=np.linspace(3.7,4.2,201)
        a[201:,4]=-.75;a[201:,5]=np.linspace(4.1,2.75,201);a[201:,6]=np.arange(1,202)*.75*10/3600
        ee,_,_=events(a,'synthetic',1.5,'test',2.77)
        self.assertEqual(len(ee),1)
        partial=a.copy();partial[201:,5]=np.linspace(4.1,3.2,201)
        self.assertEqual(len(events(partial,'synthetic',1.5,'test',2.77)[0]),0)


if __name__=='__main__':unittest.main()
