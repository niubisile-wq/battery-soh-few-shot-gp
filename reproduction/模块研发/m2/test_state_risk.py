from dataclasses import replace
import numpy as np
import pytest
import reference_gate as rg
import state_risk as sr
from test_m1 import toy


def fixture():
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(3)]
    es=[dict(cell_id=c.id,domain=c.domain,base=np.array([.98,.9,.8]),raw_base=np.array([.98,.9,.8]),
             residual=0.,raw_residual=0.,physical=np.array([.96,.88,.78]),y=np.array([.97,.89,.79])) for c in cells]
    return cells,es


def test_uniform_weight_quantile_matches_numpy_and_proxy_is_bounded():
    a=np.array([3.,1.,4.,2.])
    for q in [.2,.5,.8,.95]: assert sr.weighted_quantile(a,np.ones(4),q)==np.quantile(a,q)
    np.testing.assert_array_equal(sr.proxy(np.array([2.,.5,-2.]),np.ones(10)),[0.,.5,1.])


@pytest.mark.parametrize('kind',['raw','physical'])
def test_state_head_uses_only_support_labels_and_budget_episode_errors(kind):
    cells,es=fixture()
    a=rg.fit(es,cells,raw_reference=True,state_risk=True,risk_objective='minimax',state_proxy=kind)
    changed=[]
    for c in cells:
        y=c.y.copy();y[rg.sa.K:]=999.;changed.append(replace(c,y=y))
    b=rg.fit(es,changed,raw_reference=True,state_risk=True,risk_objective='minimax',state_proxy=kind)
    for name in ['state_centers','state_costs']: np.testing.assert_array_equal(a[name],b[name])
    assert a['state_width']==b['state_width']


def test_physical_state_uses_physical_prediction_and_preserves_query_prefix():
    cells,es=fixture();head=rg.fit(es,cells,raw_reference=True,state_risk=True,risk_objective='minimax',state_proxy='physical')
    other=[dict(e,raw_base=e['raw_base']+.1) for e in es]
    other_head=rg.fit(other,cells,raw_reference=True,state_risk=True,risk_objective='minimax',state_proxy='physical')
    np.testing.assert_array_equal(head['state_centers'],other_head['state_centers'])
    e=es[0];c=cells[0];full=rg.attach(dict(e),c,head)
    short={k:(v[:2] if isinstance(v,np.ndarray) else v) for k,v in e.items()}
    cut=rg.attach(short,rg.sa.prefix(c,rg.sa.K+2),head)
    y=c.y.copy();y[rg.sa.K:]=999.
    changed=rg.attach(dict(e,y=np.full(3,999.)),replace(c,y=y),head)
    for view,p in full['reference_predictions'].items():
        np.testing.assert_array_equal(cut['reference_predictions'][view],p[:,:2])
        np.testing.assert_array_equal(changed['reference_predictions'][view],p)


def test_state_routing_changes_with_current_proxy_and_is_prefix_causal():
    cells,es=fixture();head=rg.fit(es,cells,raw_reference=True,state_risk=True,risk_objective='minimax')
    parent=next(j for j,p in enumerate(head['params']) if p['beta']==0 and p['physical']==0)
    physical=next(j for j,p in enumerate(head['params']) if p['beta']==0 and p['physical']==1)
    head['state_centers']=np.array([0.,.2]);cost=np.full((3,2,len(head['params']),3),2.)
    cost[:,0,parent]=.1;cost[:,1,physical]=.1;head['state_costs']=cost
    c=cells[0];support=c.y[:rg.sa.K].mean();raw=support-np.array([0.,.1,.2])
    e=dict(base=raw,raw_base=raw,residual=0.,raw_residual=0.,physical=np.full(3,.6))
    result=rg.attach(dict(e),c,head)
    expected=np.array([raw[0],.5*(raw[1]+.6),.6])
    for pred in result['reference_predictions'].values(): np.testing.assert_allclose(pred,np.tile(expected,(6,1)))
    short={k:(v[:2] if isinstance(v,np.ndarray) else v) for k,v in e.items()}
    cut=rg.attach(short,rg.sa.prefix(c,rg.sa.K+2),head)
    y=c.y.copy();y[rg.sa.K:]=123.
    changed=rg.attach(dict(e,y=np.full(3,123.)),replace(c,y=y),head)
    for view,p in result['reference_predictions'].items():
        np.testing.assert_array_equal(cut['reference_predictions'][view],p[:,:2])
        np.testing.assert_array_equal(changed['reference_predictions'][view],p)
