import unittest
from types import SimpleNamespace
import numpy as np
from inspect_pl import old_table
from verify_confirmation_data import raw_clock_check,window_from_indices


class ConfirmationTests(unittest.TestCase):
    def make_table(self):
        names=['Time_sec','Date_Time','Step','Cycle','Current_Amp','Voltage_Volt','Charge_Ah','Discharge_Ah']
        vv=np.empty((1,8),dtype=object);xx=np.empty((1,8),dtype=object)
        for j,name in enumerate(names):vv[0,j]=np.array([name]);xx[0,j]=np.array([[j],[j+1]],dtype=float)
        return SimpleNamespace(classname='table',properties=dict(varnames=vv,data=xx,nrows=np.array([[2.]]),nvars=np.array([[8.]])))

    def test_named_legacy_columns(self):
        table=self.make_table();matrix,names=old_table(table)
        self.assertEqual(matrix.shape,(2,8));self.assertEqual(names[-1],'Discharge_Ah')
        np.testing.assert_array_equal(matrix[0],np.arange(8))

    def test_unrecognized_capacity_unit_blocks_decode(self):
        table=self.make_table();table.properties['varnames'][0,-1]=np.array(['Discharge_mAh'])
        with self.assertRaises(AssertionError):old_table(table)

    def test_clock_correction_cannot_change_voltage(self):
        original=np.array([[1,10,4000,1,1,4,0],[2,20,410,1,1,4,0]],dtype=float)
        corrected=original.copy();corrected[1,2]+=3600
        rule=[dict(after_row=0,added_seconds=3600)]
        raw_clock_check(original,corrected,rule)
        corrected[1,5]+=.01
        with self.assertRaises(AssertionError):raw_clock_check(original,corrected,rule)

    def test_window_reconstruction_uses_measured_seconds(self):
        voltage=np.linspace(3.845,4.145,31);a=np.zeros((31,7));a[:,1]=np.arange(31)*10;a[:,4]=.75;a[:,5]=voltage
        e=dict(raw_charge_start_index=0,raw_charge_stop_index=31,raw_window_start_index=5,window_raw_observations=22)
        x=window_from_indices(a,e)
        np.testing.assert_allclose(x[0,[0,-1]],[3.9,4.1],rtol=0,atol=1e-6)
        np.testing.assert_allclose(x[2,[0,-1]],[0,200],rtol=0,atol=1e-6)


if __name__=='__main__':unittest.main()
