import numpy as np
import nested_shrink_probe as sp

PARAMS=[dict(beta=0.,physical=0.),dict(beta=0.,physical=1.),dict(beta=.5,physical=0.)]


def test_consistent_improvement_is_retained():
    costs=np.tile(np.array([[3.,4.,5.],[4.,5.,6.],[1.,2.,3.]])[None],(6,1,1))
    c=sp.choose(costs,['a']*3+['b']*3,PARAMS,n_bags=16)
    assert c['reference']==0 and c['selected']==2
    assert c['coefficient']==1.


def test_tail_regression_blocks_mae_only_improvement():
    costs=np.tile(np.array([[3.,4.,5.],[4.,5.,6.],[1.,2.,7.]])[None],(6,1,1))
    c=sp.choose(costs,['a']*3+['b']*3,PARAMS,n_bags=16)
    assert c['selected']==2 and c['coefficient']==0.


def test_inference_ignores_query_answers():
    e=dict(base=np.array([.8,.7]),physical=np.array([.9,.8]),residual=.1,y=np.array([.7,.6]))
    choice=dict(reference=0,selected=2,coefficient=.3,bag_indices=[0,1,2])
    a=sp.predictions(e,PARAMS,choice);b=sp.predictions(dict(e,y=np.full(2,np.nan)),PARAMS,choice)
    for method in a:np.testing.assert_array_equal(a[method],b[method])


def test_domain_regression_blocks_pooled_improvement():
    costs=np.tile(np.array([[3.,4.,5.],[4.,5.,6.],[1.,2.,3.]])[None],(6,1,1))
    costs[3:,2]=[4.,5.,6.]
    c=sp.choose(costs,['a']*3+['b']*3,PARAMS,n_bags=16)
    assert c['coefficient']>0 and c['domain_coefficient']==0.
