"""Final experimental registration; verify immutable evidence, never refit/select."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from random_function_acceptance import evidence_gate

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
ROOT = DEV / 'results/m4_random_function_candidate_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    acceptance_path = ROOT / 'acceptance.json'
    selection_path = HERE / 'selected_candidate.json'
    assert not acceptance_path.exists() and not selection_path.exists()
    e = read(ROOT / 'evidence.json')
    assert e == evidence_gate(DEV / 'results')
    c = read(ROOT / 'candidate.json'); req = read(ROOT / 'request.json')
    a = read(ROOT / 'verification.json'); r = read(ROOT / 'runtime_verification.json')
    assert c['family'] == 'function_private' and c['global_step'] == .5
    assert a['status'] == 'EXPORTED_CANDIDATE_AUDIT_PASS'
    assert r['status'] == 'ISOLATED_SNAPSHOT_RUNTIME_PASS'
    assert (a['cells'], a['rows'], a['adapters'], a['edges']) == (365, 5840, 168, 32)
    assert max(a['errors'].values()) < 1e-8
    assert (r['cells'], r['adapters'], r['error']) == (3, 24, 0.)
    assert a['candidate_sha256'] == r['candidate_sha256'] == digest(ROOT / 'candidate.json')
    assert a['request_sha256'] == c['request_sha256'] == digest(ROOT / 'request.json')
    assert req['evidence_sha256'] == digest(ROOT / 'evidence.json')
    assert r['audit_sha256'] == digest(ROOT / 'verification.json')
    assert r['attempt_sha256'] == digest(ROOT / 'runtime_attempt_v2.json')
    assert r['supplement_sha256'] == digest(ROOT / 'runtime_snapshot_supplement.json')
    assert read(ROOT / 'runtime_attempt_v2.json')['exit_code'] == 0
    for name, h in req['live_code_hashes'].items(): assert digest(DEV / name) == h, name
    for name, h in req['source_snapshot'].items(): assert digest(ROOT / name) == h, name
    for name, h in a['helper_hashes'].items(): assert digest(HERE / name) == h
    assert digest(ROOT / 'audit_snapshot/verify_random_function_candidate.py') == a['audit_snapshot_sha256']
    supplement = read(ROOT / 'runtime_snapshot_supplement.json')
    assert supplement['failed_attempt_sha256'] == digest(ROOT / 'runtime_attempt.json')
    for row in supplement['files']:
        assert digest(ROOT / row['path']) == row['sha256'] == digest(row['original_path'])
    assert len(c['manifest']) == 168 and len(c['data']) == len(c['predictions']) == 365
    for row in c['manifest']: assert digest(ROOT / row['artifact']) == row['sha256']
    for row in c['predictions']: assert digest(ROOT / row['path']) == row['sha256']
    from source_oof import sa
    for row in c['data']: assert digest(sa.ROOT / row['path']) == row['sha256']
    for row in c['folds']:
        assert digest(ROOT / 'folds' / row['fold'].replace(':', '__') / 'complete.json') == row['complete_sha256']
    assert len(c['folds']) == 21
    assert digest(DEV / 'results/m3_waveform_candidate_v1/candidate.json') == e['parent_manifest_sha256']
    assert digest(DEV / 'results/m2_reference_candidate_v3/candidate.json') == e['m2_manifest_sha256']
    keys = ('mae', 'rmse', 'p95_ae')
    with (ROOT / 'cells.csv').open() as f: rows = list(csv.DictReader(f))
    assert len(rows) == len({(x['cell_id'], x['group']) for x in rows}) == 5840
    buckets = defaultdict(list); domains = defaultdict(list)
    for x in rows: buckets[x['dataset'], x['group'], x['domain']].append([float(x[k]) for k in keys])
    for (ds, g, _), values in buckets.items(): domains[ds, g].append(np.mean(values, axis=0))
    table = {k: np.mean(v, axis=0) for k, v in domains.items()}
    with (ROOT / 'comparison.csv').open() as f: comparison = list(csv.DictReader(f))
    assert len(table) == len(c['table']) == len(comparison) == 48
    for x in c['table'] + comparison:
        np.testing.assert_allclose(table[x['dataset'], x['group']], [float(x[k]) for k in keys], rtol=0, atol=1e-8)
    gates = {}
    for n in range(4):
        for sub in itertools.combinations('1234', n):
            for add in sorted(set('1234') - set(sub)):
                child = 'B' + ''.join(sorted((*sub, add))); parent = 'B' + ''.join(sub)
                gates[child + '<' + parent] = bool(all(np.all(table[ds, child] < table[ds, parent]) for ds in ('XJTU', 'MATR', 'Tongji')))
    assert gates == c['gates'] and len(gates) == 32 and all(gates.values())
    bindings = {name: digest(ROOT / name) for name in ('candidate.json', 'request.json', 'evidence.json', 'verification.json', 'runtime_verification.json', 'runtime_snapshot_supplement.json', 'cells.csv', 'comparison.csv', 'README.md')}
    report = DEV.parent / '心路历程/第四模块候选_平滑个体函数_完整消融与审核.md'
    decision = "Agent experimental selection under user's active M4-development goal; not an assertion of explicit user approval of this exact version"
    acceptance = dict(status='SELECTED_EXPERIMENTAL_M4_RELEASE', family='function_private', global_step=.5,
        decision_source=decision, evidence_sha256=bindings, report_sha256=digest(report), finalizer_sha256=digest(__file__),
        verified_scope=dict(folds=21, cells=365, adapters=168, metric_rows=5840, full_ablation_groups=16, passing_edges=32),
        risk_counts=e['risk_counts'], initialization_all32_passes=5,
        parent_selections_sha256={f'm{i}/selected_candidate.json':digest(DEV / f'm{i}/selected_candidate.json') for i in (2, 3)},
        limits=a['limits'], remaining_m4_delivery_work=[],
        historical_status_note='candidate.json and evidence.json preserve pre-audit stage statuses; this separate record registers final experimental acceptance.')
    selection = dict(status=acceptance['status'], decision_source=decision,
        candidate='模块研发/results/m4_random_function_candidate_v1', variant='function_private_global_step_0.5',
        candidate_manifest_sha256=bindings['candidate.json'], parent='m3_waveform_candidate_v1',
        acceptance='模块研发/results/m4_random_function_candidate_v1/acceptance.json',
        report='心路历程/第四模块候选_平滑个体函数_完整消融与审核.md', limits=a['limits'])
    with acceptance_path.open('x') as f: json.dump(acceptance, f, ensure_ascii=False, indent=2); f.write('\n')
    selection['acceptance_sha256'] = digest(acceptance_path)
    with selection_path.open('x') as f: json.dump(selection, f, ensure_ascii=False, indent=2); f.write('\n')
    assert read(selection_path)['acceptance_sha256'] == digest(acceptance_path)
    print('SELECTED_EXPERIMENTAL_M4_RELEASE: integrity, parents, 365 cells, 168 adapters, 16 groups, 32 edges verified', flush=True)


if __name__ == '__main__': main()
