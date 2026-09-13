"""Audit coverage, paired resampling and aggregation; not a prediction replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import repair_diagnostics as rd


def audit(root):
    root = Path(root).resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    result = json.loads((root / 'summary.json').read_text())
    assert protocol['status'] == result['status'] == 'COMPLETE'
    candidate = json.loads((Path(protocol['candidate']) / 'candidate.json').read_text())
    cells = [c for ds in rd.sa.PROTOCOL['datasets'] for c in rd.sa.load_cells(ds)]
    byid = {c.id: c for c in cells}
    seeds = protocol['seeds']
    assert len(seeds) == len(set(seeds))
    groups = ['Base+M2', 'Base+M1+M2']
    expected = {(seed, g, c.id) for seed in seeds for g in groups for c in cells}
    rows = result['rows']
    keys = [(r['seed'], r['group'], r['cell_id']) for r in rows]
    assert len(keys) == len(set(keys)) and set(keys) == expected
    bucket = defaultdict(list)
    for r in rows:
        c = byid[r['cell_id']]
        assert (r['dataset'], r['domain'], r['fold']) == (c.dataset, c.domain, c.dataset + ':' + c.domain)
        assert all(np.isfinite(r[k]) and r[k] >= 0 for k in rd.KEYS)
        bucket[r['seed'], r['dataset'], r['group']].append(r)
    aggregated = {}
    for key, rr in bucket.items():
        # Separate explicit domain means, without using the runner's macro helper.
        domains = sorted({r['domain'] for r in rr})
        aggregated[key] = {k: float(np.mean([np.mean([r[k] for r in rr if r['domain'] == d])
                                             for d in domains])) for k in rd.KEYS}
    assert len(result['summary']) == len(aggregated)
    assert len({(r['seed'], r['dataset'], r['group']) for r in result['summary']}) == len(aggregated)
    max_error = 0.
    for r in result['summary']:
        a = aggregated[r['seed'], r['dataset'], r['group']]
        for k in rd.KEYS:
            err = abs(a[k] - r[k]); max_error = max(max_error, err)
            assert err < 1e-12
    controls = {(r['dataset'], r['group']): r for r in candidate['table']}
    comparisons = [('independent', 'Base+M2', 'Base'),
                   ('incremental', 'Base+M1+M2', 'Base+M1'),
                   ('complementary', 'Base+M1+M2', 'Base+M2')]
    computed = []; failures = []
    for seed in seeds:
        gate = dict(seed=seed)
        for name, new, old in comparisons:
            gate[name] = True
            for ds in rd.sa.PROTOCOL['datasets']:
                a = aggregated[seed, ds, new]
                b = aggregated[seed, ds, old] if old in groups else controls[ds, old]
                for k in rd.KEYS:
                    delta = 100 * (a[k] - b[k])
                    if delta >= 0:
                        gate[name] = False
                        failures.append(dict(seed=seed, comparison=name, dataset=ds, metric=k, delta_pp=delta))
        computed.append(gate)
    assert computed == result['gates']
    manifests = {(m['fold'], m['group']): m for m in candidate['manifest']}
    samples = {(s['fold'], s['seed'], s['group']): s for s in result['resamples']}
    assert len(samples) == len(result['resamples']) == len(manifests) * len(seeds)
    folds = sorted({m['fold'] for m in candidate['manifest']})
    for fold in folds:
        for seed in seeds:
            pair = [samples[fold, seed, g] for g in groups]
            assert pair[0]['source_ids'] == pair[1]['source_ids']
            assert pair[0]['counts'] == pair[1]['counts']
            s = pair[0]
            assert len(s['source_ids']) == len(set(s['source_ids'])) == len(s['counts'])
            assert all(isinstance(n, int) and n >= 0 for n in s['counts'])
            mass = defaultdict(list)
            for cid, count in zip(s['source_ids'], s['counts'], strict=True):
                c = byid[cid]
                assert c.dataset == fold.split(':', 1)[0] and c.domain != fold.split(':', 1)[1]
                mass[c.domain].append(count)
            assert all(sum(v) == len(v) for v in mass.values())
    return dict(status='PASS', rows=len(rows), paired_fold_seeds=len(folds)*len(seeds),
                audit_code_sha256=rd.sa.digest(Path(__file__)),
                input_summary_sha256=rd.sa.digest(root/'summary.json'),
                max_aggregation_error=max_error,
                gate_counts={name: sum(g[name] for g in computed) for name, _, _ in comparisons},
                failures=failures,
                limits='Audits stored cell metrics, aggregation and paired source sampling; does not replay predictions or refit models. Not independent final-test evidence.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root'); args = ap.parse_args()
    answer = audit(args.root)
    rd.sa.write_json(Path(args.root) / 'aggregation_audit.json', answer)
    print(json.dumps(answer, ensure_ascii=False, indent=2))
