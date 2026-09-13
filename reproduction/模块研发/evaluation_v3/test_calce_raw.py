from datetime import datetime,timezone
import unittest
import numpy as np
from read_calce_raw import numeric_date


class DateTests(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(numeric_date(datetime(1970,1,1,0,1)),60)
        self.assertEqual(numeric_date('1970-01-01 00:01:00'),60)
        self.assertEqual(numeric_date(25569),0)
    def test_missing(self):self.assertTrue(np.isnan(numeric_date(None)))
    def test_timezone_requires_review(self):
        with self.assertRaises(ValueError):numeric_date(datetime(2000,1,1,tzinfo=timezone.utc))


if __name__=='__main__':unittest.main()
