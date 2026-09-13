import itertools
from score_source_domains import FAMILIES,group_name
from verify_source_domain_score import mapped as audit_group_name


def test_shared_parents_and_disjoint_children_cover_four_full_ablations():
    groups=['B'+''.join(s) for n in range(5) for s in itertools.combinations('1234',n)]
    parents={g for g in groups if '4' not in g}
    mapped={f:{group_name(g,f) for g in groups} for f in FAMILIES}
    assert len(groups)==16 and len(parents)==8
    assert len(set.union(*mapped.values()))==40
    for a,b in itertools.combinations(FAMILIES,2):assert mapped[a]&mapped[b]==parents
    assert group_name('B1234','cv')=='B1234'
    assert group_name('B1234','uniform')=='B1234_uniform'
    assert group_name('B1234','original_cv')=='B1234_original_cv'
    assert group_name('B1234','original_uniform')=='B1234_original_uniform'
    for family in FAMILIES:
        for group in groups:assert audit_group_name(group,family)==group_name(group,family)
