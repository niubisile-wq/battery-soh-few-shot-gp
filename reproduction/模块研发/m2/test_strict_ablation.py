"""Regression tests for the actual label and fold leakage bugs."""
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0,str(Path(__file__).resolve().parent))
import strict_ablation as sa
from test_m1 import toy
from meta_residual import predict_mode, prior


def test_pooling_other_outer_folds_is_rejected():
    rows = [{'cell_id':'valid'}]
    sa.check_validation(rows,{'valid'},{'held'})
    with pytest.raises(ValueError):
        sa.check_validation(rows+[{'cell_id':'held'}],{'valid'},{'held'})
    with pytest.raises(ValueError):
        sa.check_validation(rows*2,{'valid'},{'held'})


def test_unbudgeted_labels_are_masked_and_partitions_do_not_replenish():
    cs = [toy(i,cid=f'c{i}',domain=str(i%3)) for i in range(8)]
    allowed = np.r_[np.arange(sa.K),15,20]
    visible = [sa.source_view(c,allowed) for c in cs]
    for c in visible:
        assert np.array_equal(np.flatnonzero(np.isfinite(c.y)),allowed)
        assert np.isnan(sa.inference_view(c).y[sa.K:]).all()
    for seed in sa.PROTOCOL['seeds']:
        halves = sa.partition_source(visible,seed)
        ids = [{c.id for c in h} for h in halves]
        assert not ids[0]&ids[1]
        assert ids[0]|ids[1] == {c.id for c in cs}


def test_physical_anchor_is_preserved_in_all_prediction_modes():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        cs = [toy(i,cid=f'c{i}') for i in range(3)]
        spec = sa.SPECS['P02_anchor_physics']
        model = sa.GPModel(spec,spec['configs'][0]).fit(cs,sa.source_indices(cs))
        target = toy(8,cid='held')
        for mode in ['source_only','source_bias','posterior']:
            assert_allclose(predict_mode(model,target,mode),model.predict(target,mode),atol=1e-12)
        raw = model.gp.predict(model.features.transform(target))
        assert_allclose(prior(model,target)-raw,float(target.y[:sa.K].mean()),atol=1e-12)


def test_selector_never_accepts_test_labels():
    cal = IsotonicRegression(out_of_bounds='clip').fit([.8,1.],[.8,1.])
    v = {'cell_id':'v','dataset':'toy','domain':'d','y':np.array([.88,.92]),
         'base':np.array([.85,.93]),'residual':.01,'physical':np.array([.9,.93])}
    pick = sa.choose_parameters([v],cal,{'v'},{'test'})
    assert pick['selection_cell_ids'] == ['v']
    contaminated = dict(v,cell_id='test')
    with pytest.raises(ValueError):
        sa.choose_parameters([v,contaminated],cal,{'v'},{'test'})


def test_macro_is_cell_then_domain_weighted():
    rows = [{'dataset':'toy','domain':'a','mae':.01}]*10
    rows += [{'dataset':'toy','domain':'b','mae':.09}]
    assert_allclose(sa.macro(rows),.05)
    y = np.array([.6,.8,1.]); p = np.ones(3)
    m = sa.metrics(y,p)
    assert m['mae'] < m['rmse'] < m['p95_ae']


def test_historical_adapter_replays_physical_branch_and_new_adapter_can_fall_back():
    cal = IsotonicRegression(out_of_bounds='clip').fit([0.,1.],[0.,1.])
    base=np.array([.7,.9]); physical=np.array([.6,.8])
    old={'alpha':0.,'beta':0.,'weight_a':.25}
    assert_allclose(sa.combine(base,0.,physical,cal,old),.25*base+.75*physical)
    new=dict(old,weight_a=0.,weight_parent=1.,weight_physical=0.)
    assert_allclose(sa.combine(base,0.,physical,cal,new),base)


def test_low_soh_audit_uses_stored_value_without_float32_threshold_rounding():
    from summarize_strict import recalculate
    y=np.asarray([.90,.91],dtype=np.float32)
    assert float(y[0]) < .90  # Actual stored numeric value is 0.899999976...
    m=recalculate(y,np.asarray([.95,.94]))
    assert m['low_soh_n']==1
    assert_allclose(m['low_soh_mae'],.95-float(y[0]),atol=1e-12)
