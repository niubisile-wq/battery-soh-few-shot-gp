import unittest
import numpy as np
from partial_charge_protocol import Window, extract, support_reference


class PartialCharge(unittest.TestCase):
    def cycle(self):
        t = np.linspace(0, 20, 81) ** 1.2
        return dict(time_in_s=t, voltage_in_V=3 + .02*t,
                    current_in_A=np.ones(len(t)), cycle_number=107)

    def test_measured_time_window_and_cycle_identity(self):
        x, meta = extract(self.cycle(), Window(3.1, 3.3))
        np.testing.assert_allclose(x[0, [0, -1]], [3.1, 3.3], atol=1e-6)
        self.assertAlmostEqual(meta['duration_s'], 10)
        self.assertEqual(meta['cycle_number'], 107)
        np.testing.assert_allclose(x[2, [0, -1]], [0, 10], atol=1e-6)

    def test_never_join_across_discharge(self):
        c = self.cycle()
        c['current_in_A'][25:35] = -1
        with self.assertRaises(ValueError):
            extract(c, Window(3.1, 3.3))

    def test_time_reset_rejected(self):
        c = self.cycle()
        c['time_in_s'][30] = 0
        with self.assertRaises(ValueError):
            extract(c, Window(3.1, 3.3))

    def test_reference_is_explicit(self):
        q, policy = support_reference([1.1, 1.08, 1.09])
        self.assertEqual(q, 1.1)
        self.assertIn('fallback', policy)
        q, policy = support_reference([1.1, 1.08, 1.09], [0, 1, 2])
        self.assertAlmostEqual(q, 1.09)
        with self.assertRaises(ValueError):
            support_reference([1.1], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
