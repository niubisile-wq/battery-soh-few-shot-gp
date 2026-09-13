import itertools
from score_random_function_fit import FAMILIES, group_name
from verify_random_function_fit_score import mapped


def test_five_initializations_share_only_eight_parents():
    groups = ['B' + ''.join(c) for n in range(5) for c in itertools.combinations('1234', n)]
    sets = [{group_name(g, f) for g in groups} for f in FAMILIES]
    assert len(set.union(*sets)) == 48
    for a, b in itertools.combinations(sets, 2): assert len(a & b) == 8
    for f in FAMILIES:
        for g in groups: assert group_name(g, f) == mapped(g, f)
    assert group_name('B1234', 'init_1.0') == 'B1234'
