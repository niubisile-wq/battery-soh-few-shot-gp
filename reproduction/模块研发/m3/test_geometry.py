from dataclasses import replace
import unittest
import numpy as np
from geometry import features, GeometryGP
from screen import sa
from data import Cell
from threadpoolctl import threadpool_limits


class GeometryTests(unittest.TestCase):
    def setUp(self):
        n = 28
        t = np.tile(np.linspace(0, 900, 128), (n, 1)) * np.linspace(1, .8, n)[:, None]
        v = np.tile(np.linspace(3.8, 4.1, 128), (n, 1))
        i = np.ones_like(v)
        self.cell = Cell('synthetic', 'XJTU', 'synthetic', np.stack([v,i,t],axis=1),
                            np.linspace(1,.8,n), np.arange(n), 1., 1.)

    def test_features_prefix_and_labels(self):
        for view in ['relative', 'relative_absolute']:
            original = features(self.cell,view)
            np.testing.assert_array_equal(original[:14],features(sa.prefix(self.cell,14),view))
            np.testing.assert_array_equal(original,features(replace(self.cell,y=np.full(28,np.nan)),view))

    def test_support_reference(self):
        f = features(self.cell,'relative')
        self.assertLess(f[-1,8],f[10,8])
        self.assertEqual(f.shape,(28,11))

    def test_model_boundaries_and_budget(self):
        allowed = {'synthetic':list(range(10))+[15,20]}
        c = self.cell
        with threadpool_limits(limits=1):
            model = GeometryGP('relative','rbf').fit([sa.source_view(c,allowed[c.id])],allowed)
            p = model.predict(sa.inference_view(c))
            changed = c.y.copy(); changed[10:] = 123
            np.testing.assert_allclose(p,model.predict(replace(c,y=changed)),rtol=0,atol=1e-12)
            # Different BLAS batch shapes can change rounding (~5e-12 here).
            # Match the prospectively declared 1e-8 inference-boundary tolerance.
            np.testing.assert_allclose(p[:4],model.predict(sa.prefix(c,14)),rtol=0,atol=1e-8)
            changed = c.y.copy(); changed[11:15] = 123
            other = GeometryGP('relative','rbf').fit([sa.source_view(replace(c,y=changed),allowed[c.id])],allowed)
            np.testing.assert_allclose(p,other.predict(c),rtol=0,atol=1e-12)
            self.assertEqual(len(model.source_keys),12)


if __name__=='__main__': unittest.main()
