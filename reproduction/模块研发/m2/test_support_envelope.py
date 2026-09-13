import numpy as np
from diagnose_support_envelope import envelope
from dataclasses import replace
import reference_gate as rg
import pytest
from test_m1 import toy


def test_envelope_preserves_lower_predictions_and_prefix():
    p=np.array([.9,1.1,.8]);support=np.array([1.,.99])
    np.testing.assert_array_equal(envelope(p,support),[.9,1.,.8])
    np.testing.assert_array_equal(envelope(p,support)[:2],envelope(p[:2],support))


def test_bound_can_harm_true_recovery_so_it_is_not_guaranteed():
    p=np.array([1.01]);y=np.array([1.01]);bounded=envelope(p,np.array([1.]))
    assert abs(bounded-y)[0]>abs(p-y)[0]


def test_physical_cap_fit_matches_explicit_risk_inputs_without_mutation():
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(3)]
    es=[dict(cell_id=c.id,domain=c.domain,y=np.array([.9,.8]),base=np.array([1.1,.85]),physical=np.array([1.2,.7]),residual=.01) for c in cells]
    a=rg.fit(es,cells,physical_cap=True)
    b=rg.fit([rg.physical_envelope(e,c) for e,c in zip(es,cells)],cells)
    np.testing.assert_array_equal(a['costs'],b['costs'])
    assert es[0]['physical'][0]==1.2 and es[0]['base'][0]==1.1
    changed=[replace(c,y=np.r_[c.y[:rg.sa.K],np.full(len(c.y)-rg.sa.K,np.nan)]) for c in cells]
    d=rg.fit(es,changed,physical_cap=True)
    np.testing.assert_array_equal(a['costs'],d['costs'])
    for c,e in zip(cells,es):
        actual=rg.attach(e,c,a);expected=rg.attach(rg.physical_envelope(e,c),c,b)
        np.testing.assert_array_equal(actual['physical'],e['physical'])
        for key in actual['reference_predictions']:
            np.testing.assert_array_equal(actual['reference_predictions'][key],expected['reference_predictions'][key])


def test_raw_supported_cap_allows_corroborated_recovery_and_is_prefix_causal():
    c=toy(0)
    e=dict(physical=np.array([1.3,1.3,.8]),raw_base=np.array([1.1,.9,.9]),base=np.array([1.4,1.4,1.4]))
    result=rg.physical_envelope(e,c,'raw_supported')
    np.testing.assert_array_equal(result['physical'],[1.1,1.,.8])
    short={k:v[:2] for k,v in e.items()}
    np.testing.assert_array_equal(rg.physical_envelope(short,rg.sa.prefix(c,rg.sa.K+2),'raw_supported')['physical'],result['physical'][:2])
    changed=replace(c,y=np.r_[c.y[:rg.sa.K],np.full(len(c.y)-rg.sa.K,999.)])
    np.testing.assert_array_equal(rg.physical_envelope(e,changed,'raw_supported')['physical'],result['physical'])
    with pytest.raises(ValueError):rg.physical_envelope(dict(physical=e['physical']),c,'raw_supported')


def test_raw_supported_mode_survives_fit_and_matches_explicit_risk():
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(3)]
    es=[dict(cell_id=c.id,domain=c.domain,y=np.array([.95,.85]),base=np.array([.9,.8]),
             raw_base=np.array([1.1,.9]),physical=np.array([1.2,1.2]),residual=.01) for c in cells]
    fitted=rg.fit(es,cells,physical_cap='raw_supported')
    expected=rg.fit([rg.physical_envelope(e,c,'raw_supported') for e,c in zip(es,cells)],cells)
    assert fitted['physical_cap']=='raw_supported'
    np.testing.assert_array_equal(fitted['costs'],expected['costs'])
    a=rg.attach(es[0],cells[0],fitted)
    b=rg.attach(rg.physical_envelope(es[0],cells[0],'raw_supported'),cells[0],expected)
    for key in a['reference_predictions']:np.testing.assert_array_equal(a['reference_predictions'][key],b['reference_predictions'][key])
    np.testing.assert_array_equal(a['physical'],es[0]['physical'])
