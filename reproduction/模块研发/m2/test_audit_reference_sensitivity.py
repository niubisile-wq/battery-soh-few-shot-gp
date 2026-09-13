import json
from types import SimpleNamespace
import pytest
import audit_reference_sensitivity as ar


@pytest.fixture
def sensitivity_fixture(tmp_path, monkeypatch):
    cells = [SimpleNamespace(id='c'+d, dataset='toy', domain=d) for d in ['a','b','c']]
    monkeypatch.setattr(ar.rd.sa, 'PROTOCOL', {'datasets':['toy']})
    monkeypatch.setattr(ar.rd.sa, 'load_cells', lambda ds: cells)
    groups = ['Base+M2', 'Base+M1+M2']; seeds = [101,102]
    manifest = [dict(fold='toy:'+c.domain, group=g) for c in cells for g in groups]
    controls = [dict(dataset='toy', group=g, **{k:1. for k in ar.rd.KEYS}) for g in ['Base','Base+M1']]
    rows = [dict(seed=s, fold='toy:'+c.domain, dataset='toy', domain=c.domain,
                 cell_id=c.id, group=g, **{k:v for k in ar.rd.KEYS})
            for s in seeds for c in cells for g,v in zip(groups,[.8,.7],strict=True)]
    summary = [dict(seed=s, dataset='toy', group=g, **{k:v for k in ar.rd.KEYS})
               for s in seeds for g,v in zip(groups,[.8,.7],strict=True)]
    samples = [dict(fold='toy:'+c.domain, seed=s, group=g,
                    source_ids=[x.id for x in cells if x.id!=c.id], counts=[1,1])
               for c in cells for s in seeds for g in groups]
    gates = [dict(seed=s, independent=True, incremental=True, complementary=True) for s in seeds]
    result = dict(status='COMPLETE', rows=rows, summary=summary, resamples=samples, gates=gates)
    (tmp_path/'candidate.json').write_text(json.dumps(dict(manifest=manifest,table=controls)))
    (tmp_path/'protocol.json').write_text(json.dumps(dict(status='COMPLETE',seeds=seeds,candidate=str(tmp_path))))
    (tmp_path/'summary.json').write_text(json.dumps(result))
    return tmp_path, result


def test_sensitivity_audit_checks_full_cartesian_coverage(sensitivity_fixture):
    root, _ = sensitivity_fixture
    report = ar.audit(root)
    assert report['rows']==12 and report['paired_fold_seeds']==6
    assert report['gate_counts']==dict(independent=2,incremental=2,complementary=2)
    assert not report['failures']


@pytest.mark.parametrize('damage',['duplicate_row','wrong_gate','unpaired_counts','wrong_mean'])
def test_sensitivity_audit_rejects_inconsistent_records(sensitivity_fixture, damage):
    root, result = sensitivity_fixture
    if damage=='duplicate_row': result['rows'][0]=result['rows'][1]
    if damage=='wrong_gate': result['gates'][0]['complementary']=False
    if damage=='unpaired_counts': result['resamples'][0]['counts']=[2,0]
    if damage=='wrong_mean': result['summary'][0]['mae']+=.01
    (root/'summary.json').write_text(json.dumps(result))
    with pytest.raises(AssertionError): ar.audit(root)
