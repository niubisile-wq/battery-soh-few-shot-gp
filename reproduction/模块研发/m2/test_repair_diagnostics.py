import numpy as np
import pytest
import repair_diagnostics as rd


def test_minimax_validation_selection_considers_tail_error(monkeypatch):
    monkeypatch.setattr(rd,'candidates',lambda variant:[dict(alpha=0.,beta=0.,physical=0.,
        reference_view='toy',reference_index=i) for i in range(2)])
    e=dict(cell_id='val',domain='a',dataset='toy',y=np.zeros(10),base=np.ones(10),
           reference_predictions={'toy':np.array([[.01]*8+[1.5]*2,[.7]*10])})
    legacy=rd.select([e],'source_lodo_reference_hi',{'val'},{'test'})
    joint=rd.select([e],'source_lodo_reference_minimax_hi',{'val'},{'test'})
    assert legacy['chosen']['reference_index']==0
    assert joint['chosen']['reference_index']==1
    assert joint['chosen']['objective_worst_ratio']==pytest.approx(.7)
    e['raw_base']=np.full(10,2.)
    common=rd.select([e],'source_lodo_reference_common_minimax_hi',{'val'},{'test'})
    assert common['chosen']['reference_index']==1
    assert common['chosen']['objective_worst_ratio']==pytest.approx(.35)


def test_bounded_expert_cannot_exceed_declared_correction():
    b=np.array([.8,.9]);e=dict(base=b,calibrated=b,residual=.01,physical=np.array([-5.,5.]),parent_variance=np.full(2,.0004))
    p=dict(alpha=0.,beta=.5,physical=.75,correction_cap=.02,variance_cap=0)
    corrected=b+.005
    np.testing.assert_allclose(rd.predict(e,p),corrected+np.array([-.015,.015]))
    p.update(correction_cap=.5,variance_cap=1)
    assert np.all(abs(rd.predict(e,p)-corrected)<=.75*.5*np.sqrt(.000401)+1e-12)


def test_guard_rejects_mean_gain_that_harms_one_domain():
    def ep(cid,base,physical):
        b=np.array([base,base])
        return dict(cell_id=cid,dataset='toy',domain=cid,y=np.zeros(2),
                    base=b,calibrated=b,residual=0.,physical=np.full(2,physical))
    episodes=[ep('a',.1,.15),ep('b',.8,0.)]
    average=rd.select(episodes,'joint_mean',{'a','b'},{'test'})
    guard=rd.select(episodes,'joint_domain_guard',{'a','b'},{'test'})
    assert average['chosen']['physical']>0
    assert guard['chosen']['physical']==0
    assert guard['chosen']['violation']<=1e-12
    for e in episodes:
        np.testing.assert_allclose(rd.predict(e,guard['chosen']),e['base'])


def test_selection_rejects_outer_test_cell():
    with pytest.raises(ValueError):
        rd.select([{'cell_id':'test'}],'joint_mean',{'validation'},{'test'})


def test_inference_does_not_read_query_truth():
    e=dict(base=np.array([.9,.8]),calibrated=np.array([.85,.75]),residual=.02,
           physical=np.array([.8,.7]))
    p=dict(alpha=.5,beta=.25,physical=.25)
    expected=rd.predict(e,p)
    e['y']=np.array([123.,-456.])
    np.testing.assert_array_equal(rd.predict(e,p),expected)


def test_support_gate_prefers_better_support_expert():
    e=dict(base=np.array([.9]),calibrated=np.array([.9]),residual=0.,physical=np.array([.7]),
           parent_support_mse=.01,physical_support_mse=.0001)
    p=dict(alpha=0.,beta=0.,physical=0.,temperature=1.,gate_prior=.5)
    assert rd.predict(e,p)[0]<.71
    e['parent_support_mse'],e['physical_support_mse']=e['physical_support_mse'],e['parent_support_mse']
    assert rd.predict(e,p)[0]>.89


def test_support_score_is_query_label_and_query_signal_independent():
    from test_m1 import toy
    from dataclasses import replace
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        cs=[toy(i,cid=f'c{i}') for i in range(3)]
        spec=rd.sa.SPECS['P02_anchor_physics']
        model=rd.sa.GPModel(spec,spec['configs'][0]).fit(cs,rd.sa.source_indices(cs))
        c=toy(9,cid='held')
        changed_y=c.y.copy();changed_y[rd.sa.K:]=123.
        changed_x=c.x.copy();changed_x[rd.sa.K:]=float('nan')
        changed=replace(c,y=changed_y,x=changed_x)
        for mode in ['source_only','source_bias','posterior']:
            assert rd.support_loo_mse(model,c,mode)==rd.support_loo_mse(model,changed,mode)


def test_query_variance_is_causal_and_ignores_query_labels():
    from test_m1 import toy
    from dataclasses import replace
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        cs=[toy(i,cid=f'c{i}') for i in range(3)]
        spec=rd.sa.SPECS['P02_anchor_physics']
        model=rd.sa.GPModel(spec,spec['configs'][0]).fit(cs,rd.sa.source_indices(cs))
        c=toy(10,cid='held')
        changed_y=c.y.copy();changed_y[rd.sa.K:]=123.
        for mode in ['source_only','source_bias','posterior']:
            expected=rd.query_variance(model,c,mode)
            assert np.isfinite(expected).all() and (expected>0).all()
            np.testing.assert_allclose(rd.query_variance(model,replace(c,y=changed_y),mode),expected,atol=1e-12)
            np.testing.assert_allclose(rd.query_variance(model,rd.sa.prefix(c,rd.sa.K+3),mode),expected[:3],atol=1e-12)
