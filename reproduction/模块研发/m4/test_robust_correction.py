from dataclasses import replace
import joblib
import numpy as np
from robust_correction import RobustCorrector
from test_correction import fixture


def test_masks_prefix_and_serialization(tmp_path):
    cells, ee = fixture(); c = cells['0']; p = ee[0]['predictions']['B123']
    for loss in ('absolute', 'squared'):
        for view in ('state', 'signal'):
            m = RobustCorrector(view, loss, 10).fit(ee, cells, 'B123')
            r = m.predict(c, p)
            assert len(m.source_keys) == 40 and np.max(abs(r)) <= .05
            yy = c.y.copy(); yy[10:] = 456
            np.testing.assert_array_equal(r, m.predict(replace(c, y=yy), p))
            short = replace(c, x=c.x[:13], y=c.y[:13], cycle=c.cycle[:13])
            np.testing.assert_array_equal(r[:3], m.predict(short, p[:3]))
            path = tmp_path / (view + loss + '.joblib'); joblib.dump(m, path)
            np.testing.assert_array_equal(r, joblib.load(path).predict(c, p))


def test_nonbudget_labels_not_fitted():
    cells, ee = fixture(); masked = {}
    for cid, c in cells.items():
        yy = c.y.copy(); yy[10:] = np.nan; masked[cid] = replace(c, y=yy)
    for loss in ('absolute', 'squared'):
        a = RobustCorrector('signal', loss, 10).fit(ee, cells, 'B123')
        b = RobustCorrector('signal', loss, 10).fit(ee, masked, 'B123')
        for e in ee:
            c = cells[e['cell_id']]; p = e['predictions']['B123']
            np.testing.assert_array_equal(a.predict(c, p), b.predict(c, p))
