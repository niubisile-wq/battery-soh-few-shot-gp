import numpy as np
import diagnose_expert_capacity as dc


def test_oracle_order_and_unreachable_truth():
    e=dict(base=np.array([0.,0.,0.]),physical=np.array([1.,1.,1.]),residual=0.,y=np.array([0.,.5,2.]))
    params=[dict(beta=0.,physical=p) for p in [0.,1.]]
    r=dc.capacity(e,params)
    np.testing.assert_allclose(r['hull_oracle_mae_pp'],100/3)
    np.testing.assert_allclose(r['point_oracle_mae_pp'],50.)
    np.testing.assert_allclose(r['cell_oracle_mae_pp'],250/3)
    assert r['outside_hull_fraction']==1/3


def test_aggregation_does_not_overweight_large_domains():
    rows=[]
    for parent in ['Base','Base+M1']:
        for domain,n,value in [('a',1,1.),('b',4,3.)]:
            for _ in range(n):
                rows.append(dict(dataset='toy',parent=parent,fold='f',domain=domain,**{k:value for k in
                    ['parent_mae_pp','cell_oracle_mae_pp','point_oracle_mae_pp','hull_oracle_mae_pp','outside_hull_fraction']}))
    assert all(r['parent_mae_pp']==2. for r in dc.aggregate(rows))
