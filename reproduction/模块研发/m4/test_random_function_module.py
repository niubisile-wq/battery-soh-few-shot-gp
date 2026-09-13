from dataclasses import dataclass, replace
import joblib
import numpy as np
import pytest
from random_function_module import RandomFunctionM4
from random_function_gp import RandomFunctionGP
from random_function_branch import ConditionalFunction
from test_conditional_mixed import setup


@dataclass
class Cell:
    x: np.ndarray
    y: np.ndarray
    cycle: np.ndarray
    reference_capacity: float


class MaskCheckingParent:
    def predict(self, cell, mode=None):
        assert np.isnan(cell.y[10:]).all()
        assert mode in (None, 'frozen')
        return np.full(len(cell.x) - 10, .85)


def fixture():
    model, c = setup(); gp = model.gp
    y = gp.H_.T @ gp.y_contrast_ * gp.y_scale_
    model.gp = RandomFunctionGP(optimize=False).fit(gp.X_train_, y, [cid for cid, _ in model.source_keys])
    return model, Cell(c.x, c.y, c.cycle, c.reference_capacity)


def test_formula_masks_prefix_and_serialization(tmp_path):
    source, c = fixture()
    for mode in (None, 'frozen'):
        module = RandomFunctionM4(MaskCheckingParent(), source, .125, mode)
        before = c.y.copy(); p = module.predict(c)
        q = ConditionalFunction(source, True).predict(c)
        np.testing.assert_allclose(p, .85 + .125 * (q - .85), atol=1e-12)
        np.testing.assert_array_equal(c.y, before)
        yy = c.y.copy(); yy[10:] = 999
        np.testing.assert_array_equal(p, module.predict(replace(c, y=yy)))
        short = replace(c, x=c.x[:13], y=c.y[:13], cycle=c.cycle[:13])
        np.testing.assert_allclose(p[:3], module.predict(short), atol=1e-10)
        path = tmp_path / ('module_' + str(mode) + '.joblib'); joblib.dump(module, path)
        np.testing.assert_array_equal(p, joblib.load(path).predict(c))


def test_zero_identity_and_invalid_parent_gain():
    _, c = fixture(); p = np.full(len(c.x) - 10, .8)
    module = RandomFunctionM4(MaskCheckingParent(), None, 0.)
    result = module.predict_from_parent(c, p)
    np.testing.assert_array_equal(result, p); assert result is not p
    for gain in (-1, 2, np.nan):
        with pytest.raises(ValueError): RandomFunctionM4(None, None, gain)
    for invalid in (p[:-1], np.full_like(p, np.nan), p[:, None]):
        with pytest.raises(ValueError): module.predict_from_parent(c, invalid)
