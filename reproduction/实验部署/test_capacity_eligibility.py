import unittest
from capacity_eligibility import xjtu_capacity


class CapacityEligibility(unittest.TestCase):
    def setUp(self):
        self.cell={'nominal_capacity_in_Ah':2., 'min_voltage_limit_in_V':2.5}
        self.cycle={'attribute':'RPT', 'voltage_in_V':[4.2,3.5,2.5],
                    'current_in_A':[-1.,-1.,-1.], 'discharge_capacity_in_Ah':[0.,.8,1.9]}

    def test_measured_rpt(self):
        self.assertEqual(xjtu_capacity(self.cycle,self.cell,True),1.9)

    def test_partial_discharge_rejected_even_with_plausible_capacity(self):
        self.cycle['voltage_in_V'][-1]=3.
        with self.assertRaises(ValueError):xjtu_capacity(self.cycle,self.cell,True)

    def test_aging_labels_never_substitute_for_rpt(self):
        self.cycle['attribute']='Cycling'
        with self.assertRaises(ValueError):xjtu_capacity(self.cycle,self.cell,True)
        self.assertEqual(xjtu_capacity(self.cycle,self.cell,False),1.9)

    def test_capacity_recovery_not_removed(self):
        self.cycle['discharge_capacity_in_Ah'][-1]=2.03
        self.assertEqual(xjtu_capacity(self.cycle,self.cell,True),2.03)


if __name__=='__main__':unittest.main()
