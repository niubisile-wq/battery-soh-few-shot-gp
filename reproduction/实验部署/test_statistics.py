import unittest
import numpy as np
from derive_statistics import metrics, paired_summary, equal_domain_mean


class StatisticsTests(unittest.TestCase):
    def test_percentage_points(self):
        self.assertAlmostEqual(metrics(np.array([1.,.8]),np.array([.9,.9]))['mae'],10.)

    def test_domain_not_cell_weighted(self):
        x=[1.,1.,1.,-1.];domains=['a','a','a','b']
        result=paired_summary(x,domains,1000)
        self.assertAlmostEqual(result['delta_pp'],0.)
        self.assertEqual(result['improved_cells'],1)
        self.assertEqual(result['worse_cells'],3)
        self.assertAlmostEqual(equal_domain_mean([{'domain':d,'m':v} for d,v in zip(domains,x)],'m'),0.)

    def test_absent_low_cells(self):
        rows=[{'domain':'a','m':None},{'domain':'a','m':2.},{'domain':'b','m':4.}]
        self.assertAlmostEqual(equal_domain_mean(rows,'m'),3.)

    def test_constant_difference(self):
        r=paired_summary([-2.]*5,['a','a','b','b','b'],1000)
        for key in ['delta_pp','cell_ci_low_pp','cell_ci_high_pp','domain_ci_low_pp','domain_ci_high_pp']:
            self.assertAlmostEqual(r[key],-2.)

    def test_low_threshold_uses_stored_values_in_float64(self):
        y=np.array([np.float32(.90),np.nextafter(np.float32(.90),np.float32(1.))],dtype='float32')
        self.assertEqual((np.asarray(y,dtype=float)<.90).tolist(),[True,False])


if __name__=='__main__':unittest.main()
