import unittest
from dataclasses import replace
import numpy as np
import test_geometry
from progression import progression_features,ProgressionGP
from conditional_geometry import ConditionalGeometry
from screen import sa
from threadpoolctl import threadpool_limits


class ProgressionTests(unittest.TestCase):
    def test_features_no_future_life(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        for view in ['progress_only','progress_relative','progress_absolute']:
            f=progression_features(c,view)
            np.testing.assert_array_equal(f[:13],progression_features(sa.prefix(c,13),view))
            changed=c.cycle.copy();changed[20:]+=1000
            np.testing.assert_array_equal(f[:20],progression_features(replace(c,cycle=changed),view)[:20])
            np.testing.assert_array_equal(f,progression_features(replace(c,y=np.full(len(c.y),np.nan)),view))

    def test_model_labels_and_prefix(self):
        fixture=test_geometry.GeometryTests();fixture.setUp();c=fixture.cell
        allowed={c.id:list(range(10))+[12,15,20,25]}
        with threadpool_limits(limits=1):
            parent=ProgressionGP('progress_relative','rbf').fit([sa.source_view(c,allowed[c.id])],allowed)
            for mode in ['source_bias','posterior']:
                model=ConditionalGeometry(parent,mode);p=model.predict(sa.inference_view(c))
                y=c.y.copy();y[10:]=123
                np.testing.assert_allclose(p,model.predict(replace(c,y=y)),rtol=0,atol=1e-8)
                np.testing.assert_allclose(p[:3],model.predict(sa.prefix(c,13)),rtol=0,atol=1e-8)


if __name__=='__main__':unittest.main()
