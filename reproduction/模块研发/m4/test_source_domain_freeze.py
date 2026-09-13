import json
from pathlib import Path
import pytest
from freeze_source_domain_choices import build_manifest


@pytest.mark.parametrize('status,count',[
    ('RUNNING',152),('BATCH_HAS_TASK_FAILURES',152),('ALL_INNER_TASKS_REPLAY_AUDITED',151)])
def test_incomplete_batches_cannot_freeze(monkeypatch,status,count):
    base=Path('/virtual/results')
    payloads={str(base/'m4_source_domain_selection_v1/summary.json'):{},
        str(base/'m4_source_domain_selection_v1/verification.json'):{},
        str(base/'m4_source_domain_batch_v1/result.json'):{'status':status,'tasks':[{}]*count},
        str(base/'m4_source_domain_batch_v1/request.json'):{}}
    reads=[]
    def read(path,*args,**kwargs):
        reads.append(str(path));assert str(path) in payloads,'Unexpected downstream read'
        return json.dumps(payloads[str(path)])
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(AssertionError):build_manifest(base)
    assert len(reads)==4
