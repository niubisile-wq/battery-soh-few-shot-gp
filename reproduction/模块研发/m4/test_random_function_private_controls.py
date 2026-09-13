import itertools
from score_random_function_private_controls import FAMILIES, group_name, branch_prediction
from verify_random_function_private_controls import mapped
from test_conditional_mixed import setup
from conditional_mixed import ConditionalMixed
import numpy as np


def test_four_ablations_and_private_mode_dispatch():
    groups = ['B' + ''.join(c) for n in range(5) for c in itertools.combinations('1234', n)]
    sets = [{group_name(g, f) for g in groups} for f in FAMILIES]
    assert len(set.union(*sets)) == 40
    for a, b in itertools.combinations(sets, 2): assert len(a & b) == 8
    for f in FAMILIES:
        for g in groups: assert mapped(g, f) == group_name(g, f)
    model, c = setup()
    np.testing.assert_array_equal(branch_prediction(model, 'slope_private', c), ConditionalMixed(model, True).predict(c))
    np.testing.assert_array_equal(branch_prediction(model, 'slope_shared', c), ConditionalMixed(model, False).predict(c))
