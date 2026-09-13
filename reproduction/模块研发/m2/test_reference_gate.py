from dataclasses import replace
import numpy as np
import pytest
import reference_gate as rg
from test_m1 import toy


def test_position_risk_integrates_estimated_curve_not_sparse_point_frequencies():
    q=rg.sa.K+np.arange(0,61,10)
    e=dict(base=np.array([0.,1.,0.,1.,0.,1.,0.]),residual=0.,physical=np.zeros(7),y=np.zeros(7))
    p=dict(beta=0.,physical=0.)
    result=rg.expert_risk(e,p,q,61)
    dense=np.interp(np.arange(61),q-rg.sa.K,e['base'])
    m=rg.sa.metrics(np.zeros(61),dense)
    np.testing.assert_allclose(result,[m[k] for k in ['mae','rmse','p95_ae']])
    assert result[2]<rg.expert_risk(e,p)[2]
    with pytest.raises(ValueError): rg.expert_risk(e,p,q+1,61)


def test_position_risk_never_reads_unbudgeted_source_labels(monkeypatch):
    c=toy(4,cid='a',domain='a');q=np.array([rg.sa.K,(rg.sa.K+len(c.y)-1)//2,len(c.y)-1])
    allowed=np.r_[np.arange(rg.sa.K),q]
    monkeypatch.setattr(rg.sa,'source_indices',lambda cells:{'a':allowed})
    e=dict(cell_id='a',domain='a',base=c.y[q]+np.array([.1,-.1,.1]),residual=.01,
           physical=c.y[q]+.02,y=c.y[q])
    a=rg.fit([e],[c],position_risk=True)
    y=c.y.copy();mask=np.ones(len(y),dtype=bool);mask[allowed]=False;y[mask]=999.
    b=rg.fit([e],[replace(c,y=y)],position_risk=True)
    np.testing.assert_array_equal(a['costs'],b['costs'])
    assert a['risk_sampling']=={'a':dict(query_indices=q.tolist(),n_query=len(c.x)-rg.sa.K)}


def test_common_risk_reference_uses_same_metric_scales_for_both_backbones():
    params=[dict(beta=0.,physical=0.),dict(beta=0.,physical=0.,raw_mix=1.),
            dict(beta=.5,physical=.5),dict(beta=1.,physical=.5)]
    risks=np.array([[1.,2.,4.],[2.,2.,2.],[.6,1.5,3.2],[.9,1.3,2.]])
    assert rg.risk_indices(risks,params,'minimax')==2
    assert rg.risk_indices(risks,params,'minimax',reference_index=1)==3
    np.testing.assert_array_equal(rg.risk_indices(np.stack([risks,risks]),params,'minimax',1),[3,3])


def test_common_scale_fits_and_persists_pure_raw_reference_index():
    c=toy(4,cid='a',domain='a')
    e=dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.75,.65]),
           residual=.01,raw_base=np.array([.8,.72]),raw_residual=-.01,y=np.array([.8,.72]))
    h=rg.fit([e],[c],raw_reference=True,risk_objective='minimax',common_scale=True)
    p=h['params'][h['risk_reference_index']]
    assert p==dict(beta=0.,physical=0.,raw_mix=1.)
    np.testing.assert_array_equal(h['costs'][:,h['risk_reference_index']],np.zeros((1,3)))
    same=dict(e,raw_base=e['base'],raw_residual=e['residual'])
    hh=rg.fit([same],[c],raw_reference=True,risk_objective='minimax',common_scale=True)
    assert hh['params'][hh['risk_reference_index']]==dict(beta=0.,physical=0.,raw_mix=0.)


def test_raw_reference_can_recover_raw_backbone_without_query_labels():
    c=toy(4,cid='a',domain='a')
    e=dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.95,.95]),
           residual=0.,raw_base=np.array([.7,.65]),raw_residual=0.,y=np.array([.7,.65]))
    head=rg.fit([e],[c],raw_reference=True,risk_objective='minimax')
    assert len(head['params'])==120
    p=rg.attach(dict(e),c,head)['reference_predictions']['hi']
    np.testing.assert_allclose(p,np.tile(e['raw_base'],(6,1)))
    changed=replace(c,y=np.r_[c.y[:rg.sa.K],np.full(len(c.y)-rg.sa.K,999.)])
    np.testing.assert_array_equal(rg.attach(dict(e,y=np.array([999.,999.])),changed,head)['reference_predictions']['hi'],p)


def test_identical_raw_backbone_preserves_independent_m2_risks_exactly():
    c=toy(4,cid='a',domain='a')
    e=dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.7,.65]),
           residual=.01,y=np.array([.8,.75]))
    legacy=rg.fit([e],[c],risk_objective='minimax')
    raw=rg.with_raw_reference([e],[e]);head=rg.fit(raw,[c],raw_reference=True,risk_objective='minimax')
    assert len(head['params'])==len(legacy['params'])==40
    np.testing.assert_array_equal(head['costs'],legacy['costs'])
    a=rg.attach(dict(e),c,legacy);b=rg.attach(dict(raw[0]),c,head)
    for view in a['reference_predictions']:
        np.testing.assert_array_equal(a['reference_predictions'][view],b['reference_predictions'][view])
    with pytest.raises(AssertionError): rg.with_raw_reference([e],[dict(e,y=e['y']+.1)])


def test_raw_reference_bagging_matches_explicit_expert_average():
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(5)]
    es=[dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.75,.65]),
             residual=.02,raw_base=np.array([.8,.72]),raw_residual=-.01,y=np.array([.81,.71])) for c in cells]
    h=rg.fit(es,cells,metric=True,bagged=True,n_bags=5,raw_reference=True,risk_objective='minimax')
    e=es[0];r=rg.attach(dict(e),cells[0],h)
    for i,choices in enumerate(r['reference_choices']['hi_pca8_bagged']):
        explicit=np.mean([rg.expert_prediction(e,h['params'][j]) for j in choices],axis=0)
        np.testing.assert_allclose(r['reference_predictions']['hi_pca8_bagged'][i],explicit,atol=1e-14)


def test_backbone_coverage_keeps_selected_backbone_at_low_transfer_confidence():
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(5)]
    es=[dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.75,.65]),
             residual=0.,raw_base=np.array([.8,.72]),raw_residual=0.,y=np.array([.8,.72])) for c in cells]
    h=rg.fit(es,cells,metric=True,bagged=True,n_bags=5,support_gate=True,
             raw_reference=True,backbone_coverage=True,risk_objective='minimax')
    h['views']['hi_pca8']['support_thresholds'][:]=1e-12
    target=toy(100,cid='target',domain='target')
    result=rg.attach(dict(es[0]),target,h)
    bounded=result['reference_predictions']['hi_pca8_bagged_supported']
    np.testing.assert_allclose(bounded,np.tile(es[0]['raw_base'],(18,1)),atol=1e-12)
    old=rg.attach(dict(es[0]),target,dict(h,backbone_coverage=False))
    np.testing.assert_allclose(old['reference_predictions']['hi_pca8_bagged_supported'],
                               np.tile(es[0]['base'],(18,1)),atol=1e-10)


def test_backbone_coverage_is_exact_noop_for_identical_raw_parent():
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(5)]
    es=[dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.75,.65]),
             residual=.01,y=np.array([.8,.72])) for c in cells]
    old=rg.fit(es,cells,metric=True,bagged=True,n_bags=5,support_gate=True,risk_objective='minimax')
    enriched=rg.with_raw_reference(es,es)
    new=rg.fit(enriched,cells,metric=True,bagged=True,n_bags=5,support_gate=True,risk_objective='minimax',
               raw_reference=True,backbone_coverage=True)
    a=rg.attach(dict(es[0]),cells[0],old);b=rg.attach(dict(enriched[0]),cells[0],new)
    for view in a['reference_predictions']:
        np.testing.assert_array_equal(a['reference_predictions'][view],b['reference_predictions'][view])


def test_minimax_risk_does_not_trade_tail_regression_for_mae_gain():
    params=[dict(beta=0.,physical=0.),dict(beta=.5,physical=0.),dict(beta=0.,physical=1.)]
    risk=np.array([[1.,1.,2.],[.7,.9,2.4],[.8,.9,1.8]])
    assert rg.risk_indices(risk,params)==1
    assert rg.risk_indices(risk,params,'minimax')==2
    np.testing.assert_array_equal(rg.risk_indices(np.stack([risk,risk]),params,'minimax'),[2,2])
    assert rg.risk_indices(risk*np.array([10.,.1,100.]),params,'minimax')==2


def test_legacy_risk_order_is_preserved_including_exact_ties():
    rng=np.random.default_rng(42)
    risk=rng.integers(0,3,size=(20,40,3)).astype(float)
    expected=[min(range(40),key=lambda i:tuple(r[i])) for r in risk]
    np.testing.assert_array_equal(rg.risk_indices(risk,rg.PARAMS),expected)


@pytest.mark.parametrize('objective',['mae','minimax'])
def test_reference_risk_uses_matching_source_regime_and_ignores_query_truth(objective):
    a=toy(1,cid='a',domain='a');b=toy(2,cid='b',domain='b')
    def episode(c,physical_wins):
        return dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.7,.65]),
                    residual=0.,y=np.array([.7,.65]) if physical_wins else np.array([.9,.85]))
    es=[episode(a,True),episode(b,False)];head=rg.fit(es,[a,b],risk_objective=objective)
    e=dict(es[0]);e.pop('y')
    out=rg.attach(e,a,head)['reference_predictions']['hi'][1]  # bandwidth .25, rho=1
    np.testing.assert_allclose(out,e['physical'])
    y=a.y.copy();y[rg.sa.K:]=123.;x=a.x.copy();x[rg.sa.K:]=np.nan
    changed=replace(a,y=y,x=x)
    np.testing.assert_array_equal(rg.attach(dict(e,y=np.array([123.,456.])),changed,head)['reference_predictions']['hi'][1],out)


@pytest.mark.parametrize('objective',['mae','minimax'])
def test_reference_gate_prediction_prefix_is_causal(objective):
    a=toy(1,cid='a',domain='a');e=dict(cell_id='a',domain='a',base=np.array([.9,.85]),
        physical=np.array([.7,.65]),residual=0.,y=np.array([.8,.75]))
    head=rg.fit([e],[a],risk_objective=objective);p=rg.attach(dict(e),a,head)['reference_predictions']
    short={k:(v[:1] if isinstance(v,np.ndarray) else v) for k,v in e.items()}
    q=rg.attach(short,a,head)['reference_predictions']
    for view in p: np.testing.assert_array_equal(p[view][:,:1],q[view])


def test_soft_risk_weights_are_normalized_and_replayable():
    a=toy(3,cid='a',domain='a');e=dict(cell_id='a',domain='a',base=np.array([.9,.85]),
        physical=np.array([.7,.65]),residual=.01,y=np.array([.8,.75]))
    head=rg.fit([e],[a],soft=True)
    result=rg.attach(dict(e),a,head)
    for view in ['hi_soft','physics_soft']:
        assert result['reference_predictions'][view].shape==(18,2)
        for i,p in enumerate(result['reference_choices'][view]):
            assert p['parent']>=0 and p['physical']>=0
            np.testing.assert_allclose(p['parent']+p['physical'],1.,atol=1e-12)
            np.testing.assert_allclose(result['reference_predictions'][view][i],
                p['parent']*e['base']+p['residual']*e['residual']+p['physical']*e['physical'],atol=1e-12)


def test_reference_projection_uses_source_fit_and_ignores_query_suffix():
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(5)]
    es=[dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.7,.65]),
        residual=.01,y=np.array([.8,.75])) for c in cells]
    head=rg.fit(es,cells,metric=True)
    target=toy(20,cid='held');e=dict(es[0]);e.pop('y')
    full=rg.attach(dict(e),target,head)
    x=target.x.copy();x[rg.sa.K:]=np.nan
    changed=rg.attach(dict(e),replace(target,x=x),head)
    for view in ['hi_pruned','hi_pca4','hi_pca8']:
        assert full['reference_predictions'][view].shape==(6,2)
        np.testing.assert_array_equal(full['reference_predictions'][view],changed['reference_predictions'][view])


def test_bagged_risk_preserves_domain_mass_and_averages_selected_predictions():
    cells=[toy(i,cid=str(i),domain=str(i%2)) for i in range(6)]
    es=[dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.7,.65]),
        residual=.01,y=np.array([.8,.75])) for c in cells]
    head=rg.fit(es,cells,metric=True,bagged=True)
    weights=head['risk_bootstrap_weights']
    np.testing.assert_allclose(weights.sum(1),1.)
    for domain in ['0','1']:
        ix=np.array(head['source_domains'])==domain
        np.testing.assert_allclose(weights[:,ix].sum(1),.5)
    e=es[0];r=rg.attach(dict(e),cells[0],head)
    for i,indices in enumerate(r['reference_choices']['hi_pca8_bagged']):
        preds=[]
        for j in indices:
            p=head['params'][j]
            preds.append((1-p['physical'])*(e['base']+p['beta']*e['residual'])+p['physical']*e['physical'])
        np.testing.assert_allclose(r['reference_predictions']['hi_pca8_bagged'][i],np.mean(preds,axis=0),atol=1e-12)


def test_unsupported_reference_shrinks_correction_toward_parent():
    cells=[toy(i,cid=str(i),domain=str(i%2)) for i in range(6)]
    es=[dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.7,.65]),
        residual=0.,y=np.array([.7,.65])) for c in cells]
    h=rg.fit(es,cells,metric=True,bagged=True,support_gate=True)
    e=es[0];near=rg.attach(dict(e),cells[0],h)
    assert all(p['confidence']==1 for p in near['reference_choices']['hi_pca8_bagged_supported'])
    x=cells[0].x.copy();x[:rg.sa.K]*=100
    far=rg.attach(dict(e),replace(cells[0],x=x),h)
    choices=far['reference_choices']['hi_pca8_bagged_supported']
    assert all(0<=p['confidence']<.01 for p in choices)
    for j,p in enumerate(choices):
        expected=e['base']+p['confidence']*(far['reference_predictions']['hi_pca8_bagged'][p['bagged_index']]-e['base'])
        np.testing.assert_allclose(far['reference_predictions']['hi_pca8_bagged_supported'][j],expected)


def test_consistent_source_improvement_can_override_geometric_fallback():
    cells=[toy(i,cid=str(i),domain=str(i%2)) for i in range(6)]
    es=[dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.85]),physical=np.array([.7,.65]),
        residual=0.,y=np.array([.7,.65])) for c in cells]
    head=rg.fit(es,cells,metric=True,bagged=True,support_gate=True,consensus_gate=True)
    x=cells[0].x.copy();x[:rg.sa.K]*=100
    e=dict(es[0]);e['y']=np.array([123.,456.])
    r=rg.attach(e,replace(cells[0],x=x),head)
    for p in r['reference_choices']['hi_pca8_bagged_supported_consensus']:
        assert p['source_consensus_floor']==1. and p['confidence']==1.
    np.testing.assert_allclose(r['reference_predictions']['hi_pca8_bagged_supported_consensus'],np.tile(es[0]['physical'],(18,1)))


def test_more_bags_extend_the_same_seed_sequence():
    h=dict(source_domains=['a','a','b','b'],domain_sizes={'a':2,'b':2},weight=np.full(4,.25),bagged_seeds=list(range(24)))
    original=rg.bootstrap_weights(h)
    extended=rg.bootstrap_weights(dict(h,bagged_seeds=list(range(128))))
    np.testing.assert_array_equal(extended[:24],original)
    np.testing.assert_allclose(extended.sum(1),1.)
