import itertools
from score_random_function import FAMILIES, group_name
from verify_random_function_score import mapped


def test_five_full_ablations_only_share_eight_parents():
    base = {'B' + ''.join(c) for n in range(5) for c in itertools.combinations('1234', n)}
    parents = {g for g in base if '4' not in g}
    sets = [{group_name(g, f) for g in base} for f in FAMILIES]
    assert len(parents) == 8 and all(len(s) == 16 for s in sets)
    assert len(set.union(*sets)) == 48
    for a, b in itertools.combinations(sets, 2): assert a & b == parents
    for f in FAMILIES:
        for g in base: assert group_name(g, f) == mapped(g, f)
    assert group_name('B1234', 'function_uniform') == 'B1234'
    assert group_name('B1234', 'matched_slope') == 'B1234_matched_slope'
