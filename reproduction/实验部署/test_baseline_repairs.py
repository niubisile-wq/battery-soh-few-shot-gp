import unittest
import numpy as np
import torch
import run_formal_protocol as formal
from degradation_baselines import exponential_trend


class BaselineRepairs(unittest.TestCase):
    def test_exponential_with_irregular_observations(self):
        t = np.array([0., 5., 11., 19.])
        q = np.array([21., 39.])
        pred = exponential_trend(t, 1.02 * np.exp(-.003 * t), q)
        np.testing.assert_allclose(pred, 1.02 * np.exp(-.003 * q), rtol=1e-10)

    def test_invalid_support_fails(self):
        with self.assertRaises(ValueError):
            exponential_trend([0, 1, 2], [1, 0, .9], [3])

    def test_complete_blocks_and_frozen_earlier_layers(self):
        torch.set_num_threads(1)
        for name, ctor in formal.NN_MODELS.items():
            with self.subTest(model=name):
                original = ctor()
                fitted = formal.adapt(original, np.ones((3, 3, 128), dtype=np.float32),
                                      np.full(3, .8, dtype=np.float32), "last_block",
                                      torch.device("cpu"), 1)
                before = dict(original.named_parameters())
                changed = []
                for n, p in fitted.named_parameters():
                    if not p.requires_grad:
                        self.assertTrue(torch.equal(before[n], p), n)
                    elif not torch.equal(before[n], p):
                        changed.append(n)
                self.assertTrue(changed)
                if hasattr(fitted, "rnn"):
                    for n, p in fitted.rnn.named_parameters():
                        self.assertEqual(p.requires_grad, n.endswith(f"_l{fitted.rnn.num_layers-1}"))
                if hasattr(fitted, "t"):
                    self.assertTrue(all(p.requires_grad for p in fitted.t.layers[-1].parameters()))
                    self.assertTrue(all(not p.requires_grad for p in fitted.t.layers[0].parameters()))


if __name__ == "__main__":
    unittest.main()
