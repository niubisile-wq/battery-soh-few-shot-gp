import unittest
import numpy as np
from evaluate import error_metrics, aggregate, quality_masks


class DiagnosisTests(unittest.TestCase):
    def test_bias_and_tail(self):
        m = error_metrics(np.zeros(100), np.ones(100)*.02)
        self.assertAlmostEqual(m['mae'], 2)
        self.assertAlmostEqual(m['bias_fraction'], 1)
        self.assertAlmostEqual(m['top5_error_share'], .05)
        self.assertAlmostEqual(error_metrics([0, 0], [.01, -.01])['bias_fraction'], 0)

    def test_equal_domains(self):
        rows = [dict(dataset='x', domain=d, **error_metrics([0], [v]))
                for d, v in [('a', .01), ('a', .01), ('b', .05)]]
        self.assertAlmostEqual(aggregate(rows, ['dataset'])[0]['mae'], 3)

    def test_quality_label_free(self):
        x = np.ones((12, 3, 128))
        x[:, 0] = np.linspace(3.8, 4, 128)
        x[:, 2] = np.linspace(0, 100, 128)
        masks = quality_masks(x, 10)
        self.assertTrue(masks['duration_reference_range'].all())
        self.assertTrue(masks['current_stable'].all())
        self.assertTrue(masks['voltage_monotone'].all())


if __name__ == '__main__':
    unittest.main()
