import numpy as np
import nested_binary_stability as bs


def test_binary_vote_selects_consistent_winner():
    params=[dict(beta=0.,physical=0.),dict(beta=0.,physical=1.)]
    costs=np.tile(np.array([[3.,4.,5.],[1.,2.,3.]])[None],(4,1,1))
    assert bs.binary_weight(costs,['a','a','b','b'],params)==1.
    assert bs.binary_weight(costs[:,::-1],['a','a','b','b'],params)==0.


def test_exact_metric_difference_decomposition():
    rows=[dict(seed=1,parent='Base',method=method,**{k:v for k in bs.ss.sp.METRICS}) for method,v in
          [('fixed_simple',1.),('simple',3.),('domain_guarded',2.5)]]
    for r in bs.decomposition(rows):
        assert (r['switch'],r['correction'],r['total'])==(2.,-.5,1.5)
