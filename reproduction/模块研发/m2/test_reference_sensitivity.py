from types import SimpleNamespace
import numpy as np
import pytest
from reference_sensitivity import reweight,risk_seeds


def test_fresh_risk_seeds_are_explicit_and_leave_legacy_default_unchanged():
    assert risk_seeds()==list(range(101,111))
    fresh=risk_seeds(1001,30)
    assert len(fresh)==30 and not set(fresh)&set(risk_seeds())
    with pytest.raises(ValueError): risk_seeds(-1,10)
    with pytest.raises(ValueError): risk_seeds(1001,0)


def test_nested_risk_resampling_preserves_domains_and_original_source_rows():
    domains=['a','a','a','b','b','b'];ids=[f'c{i}' for i in range(6)]
    byid={cid:SimpleNamespace(domain=d) for cid,d in zip(ids,domains,strict=True)}
    head=dict(source_ids=ids,source_domains=domains,domain_sizes={'a':3,'b':3},
        bagged_seeds=list(range(5)),weight=np.full(6,1/6),costs=np.arange(6*4*3).reshape(6,4,3),
        views={'hi_pca8':dict(z=np.arange(12).reshape(6,2),support_z=np.arange(12).reshape(6,2).copy(),support_radius=np.ones(6))})
    h,counts=reweight(head,byid,103)
    assert counts==reweight(head,byid,103)[1]
    assert sum(counts[:3])==sum(counts[3:])==3
    active=np.asarray(counts)>0
    np.testing.assert_array_equal(h['costs'],head['costs'][active])
    np.testing.assert_array_equal(h['views']['hi_pca8']['z'],head['views']['hi_pca8']['z'][active])
    np.testing.assert_array_equal(h['views']['hi_pca8']['support_z'],head['views']['hi_pca8']['support_z'])
    np.testing.assert_array_equal(h['views']['hi_pca8']['support_radius'],head['views']['hi_pca8']['support_radius'])
    assert set(h['source_ids'])<=set(ids)
    for d in ['a','b']:
        mask=np.asarray(h['source_domains'])==d
        np.testing.assert_allclose(h['weight'][mask].sum(),.5)
        np.testing.assert_allclose(h['risk_bootstrap_weights'][:,mask].sum(1),.5)
    head['state_costs']=np.stack([head['costs'],head['costs']*2],axis=1)
    dynamic,_=reweight(head,byid,103)
    np.testing.assert_array_equal(dynamic['state_costs'],head['state_costs'][active])
