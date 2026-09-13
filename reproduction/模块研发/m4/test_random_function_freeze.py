import json
from pathlib import Path
import pytest
from freeze_random_function import build_manifest


@pytest.mark.parametrize('status,count', [
    ('RUNNING', 21), ('BATCH_HAS_TASK_FAILURES', 21), ('ALL_VALIDATION_REPLAY_AUDITED', 20)])
def test_incomplete_validation_never_reads_downstream(monkeypatch, status, count):
    base = Path('/virtual/results')
    payloads = {
        str(base / 'm4_random_function_batch_v1/result.json'): dict(status=status, folds=[{}] * count),
        str(base / 'm4_random_function_batch_v1/request.json'): {}}
    reads = []
    def read(path, *args, **kwargs):
        reads.append(str(path))
        assert str(path) in payloads, 'Unexpected downstream read before batch completion'
        return json.dumps(payloads[str(path)])
    monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(AssertionError):
        build_manifest(base)
    assert len(reads) == 2
