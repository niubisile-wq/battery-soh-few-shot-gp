import unittest
import numpy as np
from reestimated_screen import augmented


class ReestimatedTests(unittest.TestCase):
    def test_zero_identity_and_preserve_calibration(self):
        e=dict(base=np.array([.8,.7]),residual=.02,physical=np.array([.81,.72]),y=np.array([.79,.71]))
        branch=np.array([.77,.68])
        np.testing.assert_array_equal(augmented(e,branch,0)['base'],e['base'])
        changed=augmented(e,branch,.5)
        np.testing.assert_allclose(changed['base'],.5*(e['base']+branch))
        self.assertIs(changed['physical'],e['physical'])
        self.assertEqual(changed['residual'],e['residual'])
        np.testing.assert_array_equal(e['base'],[.8,.7])


if __name__=='__main__':unittest.main()
