import unittest
import numpy as np
from build_calce_cohort import counter_increment,positive_runs,events_from_book


class EventTests(unittest.TestCase):
    def test_cumulative_counter(self):
        q,n=counter_increment([5.,5.,5.1,5.8,6.]);self.assertAlmostEqual(q,1);self.assertEqual(n,0)
    def test_reset_counter(self):
        q,n=counter_increment([5.,.01,.1,.8,1.]);self.assertAlmostEqual(q,1);self.assertEqual(n,1)
    def test_negative_invalid(self):
        with self.assertRaises(ValueError):counter_increment([0,-.1])
    def test_runs_ignore_cycle_id(self):
        self.assertEqual(positive_runs([0,1,1,0,-1,0,1,1,0],.01),[(1,3),(6,8)])
    def test_timer_reset_splits_observed_sessions(self):
        a=np.zeros((50,7));a[:,0]=np.arange(50);a[:,1]=np.arange(50)
        a[:,2]=1e9+np.arange(50);a[:,3]=1
        a[:30,4]=.55;a[30:,4]=-.55
        a[:30,5]=np.linspace(3.8,4.2,30);a[30:,5]=np.linspace(4.2,2.7,20)
        a[:,6]=2;a[30:,6]+=np.arange(1,21)*.55/3600
        b=a.copy();b[:,2]+=86400;b[:,0]+=50
        events,rejections,issues=events_from_book(np.r_[a,b],'test',1.1,'test.xlsx')
        self.assertEqual(len(events),2);self.assertFalse(issues)
        self.assertEqual(rejections['instrument_timer_session_boundaries'],1)
        for e in events:self.assertAlmostEqual(e['capacity_Ah'],20*.55/3600)
        self.assertEqual(events[1]['raw_charge_start_index'],50)
    def test_unobserved_gap_breaks_charge(self):
        a=np.zeros((50,7));a[:,0]=np.arange(50);a[:,1]=np.arange(50)
        a[:,2]=1e9+np.arange(50);a[15:,2]+=3600;a[:,3]=1
        a[:30,4]=.55;a[30:,4]=-.55
        a[:30,5]=np.linspace(3.8,4.2,30);a[30:,5]=np.linspace(4.2,2.7,20)
        a[:,6]=2;a[30:,6]+=np.arange(1,21)*.55/3600
        events,rejections,issues=events_from_book(a,'test',1.1,'test.xlsx')
        self.assertFalse(events);self.assertFalse(issues)
        self.assertEqual(rejections['wallclock_pause_boundaries'],1)


if __name__=='__main__':unittest.main()
