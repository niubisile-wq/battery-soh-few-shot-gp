"""Load original eight frozen parent chains, preserving their prediction modes."""
import json
import joblib
from source_oof import sa, HERE, FROZEN


def load_parents(fold):
    m3root = HERE.parent / 'results/m3_waveform_candidate_v1'
    m2 = json.loads((FROZEN / 'candidate.json').read_text())
    m3 = json.loads((m3root / 'candidate.json').read_text())
    entries = {r['group']:r for r in m2['manifest'] if r['fold'] == fold}
    records = []; adapters = {}
    for group in ('Base+M2', 'Base+M1+M2'):
        entry = entries[group]; path = FROZEN / entry['artifact']
        assert sa.digest(path) == entry['sha256']
        adapters[group] = joblib.load(path)
        records.append(dict(path=str(path), sha256=entry['sha256']))
    a0, a1 = adapters['Base+M2'], adapters['Base+M1+M2']
    budget = {tuple(k) for k in entries['Base+M2']['source_keys']}
    assert budget == {tuple(k) for k in entries['Base+M1+M2']['source_keys']} and len(budget) <= 1000
    parents = {'B':(a0.parent, a0.parent_mode), 'B1':(a1.parent, a1.parent_mode), 'B2':(a0, None), 'B12':(a1, None)}
    waveform = {r['group']:r for r in m3['manifest'] if r['fold'] == fold}
    assert set(waveform) == {'B3', 'B13', 'B23', 'B123'}
    for group, entry in waveform.items():
        path = m3root / entry['artifact']; assert sa.digest(path) == entry['sha256']
        assert {tuple(k) for k in entry['source_keys']} == budget
        parents[group] = (joblib.load(path), None)
        records.append(dict(path=str(path), sha256=entry['sha256']))
    return parents, budget, dict(m2_manifest_sha256=sa.digest(FROZEN / 'candidate.json'),
        m3_manifest_sha256=sa.digest(m3root / 'candidate.json'), artifacts=records)
