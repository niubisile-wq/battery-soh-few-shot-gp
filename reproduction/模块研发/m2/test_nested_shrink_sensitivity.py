from collections import Counter
import numpy as np
import nested_shrink_sensitivity as ss


def test_sampling_preserves_each_domain_count_and_is_repeatable():
    domains=np.array(['a','b','a','b','b','a','b'])
    a=ss.sample_rows(domains,2001);b=ss.sample_rows(domains,2001)
    np.testing.assert_array_equal(a,b)
    assert Counter(domains[a])==Counter(domains)
    assert set(a)<=set(range(len(domains)))


def test_sensitivity_aggregation_uses_equal_domains():
    rows=[]
    for domain,n,value in [('a',1,1.),('b',4,3.)]:
        rows.extend([dict(seed=1,parent='Base',method='simple',fold='f',domain=domain,**{k:value for k in ss.sp.METRICS}) for _ in range(n)])
    result=ss.summarize(rows)
    assert len(result)==1
    assert all(result[0][k]==2. for k in ss.sp.METRICS)
