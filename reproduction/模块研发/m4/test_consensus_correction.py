from dataclasses import replace
import joblib
import numpy as np
from consensus_correction import ConsensusCorrector
from test_correction import fixture


def test_exclusion_and_masked_prefix(tmp_path):
    cells, episodes = fixture()
    c = cells['0']; p = episodes[0]['predictions']['B123']
    for view in ('state', 'signal'):
        model = ConsensusCorrector(view, 'ridge', 10).fit(episodes, cells, 'B123')
        for member, spec in zip(model.members, model.exclusions):
            assert all(cells[cid].domain != spec['held_domain'] for cid, _ in member.source_keys)
            assert {cid for cid, _ in member.source_keys} == set(spec['training_ids'])
        expected = model.predict(c, p)
        yy = c.y.copy(); yy[10:] = 456
        np.testing.assert_allclose(expected, model.predict(replace(c, y=yy), p))
        short = replace(c, x=c.x[:13], y=c.y[:13], cycle=c.cycle[:13])
        np.testing.assert_allclose(expected[:3], model.predict(short, p[:3]))
        assert np.max(abs(expected)) <= .05
        path = tmp_path / (view + '.joblib'); joblib.dump(model, path)
        np.testing.assert_array_equal(expected, joblib.load(path).predict(c, p))


class Fixed:
    def __init__(self, value): self.value = value
    def predict(self, cell, pred, indices=None): return np.full(len(pred), self.value)


def test_consensus_arithmetic():
    m = ConsensusCorrector('state', 'ridge', 10)
    m.members = [Fixed(.03), Fixed(-.01)]
    mean, gate = m.components(None, np.zeros(2))
    np.testing.assert_allclose(mean, .01)
    np.testing.assert_allclose(gate, .5)
    np.testing.assert_allclose(m.predict(None, np.zeros(2)), .005)
    m.members = [Fixed(.01), Fixed(.03)]
    np.testing.assert_allclose(m.predict(None, np.zeros(2)), .02)
    m.members = [Fixed(0), Fixed(0)]
    np.testing.assert_array_equal(m.predict(None, np.zeros(2)), np.zeros(2))
