"""All five private-function global steps under ten cached paired risk draws."""
import json
from pathlib import Path
import numpy as np
from source_oof import sa, HERE
from score import PARENTS, addition_edges
from evaluate import aggregate, error_metrics, dump_csv


def main():
    results = HERE.parent / 'results'; source = results / 'm4_random_function_screen_v1'
    sensitivity = results / 'm4_random_function_sensitivity_v1'; risk = results / 'm3_waveform_stability_v2'
    audit = json.loads((source / 'verification.json').read_text())
    assert audit['status'] == 'PASS' and audit['summary_sha256'] == sa.digest(source / 'summary.json')
    check = json.loads((sensitivity / 'verification.json').read_text())
    assert check['status'] == 'PASS' and check['summary_sha256'] == sa.digest(sensitivity / 'summary.json')
    rp = json.loads((risk / 'protocol.json').read_text())
    assert rp['status'] == 'COMPLETE' and rp['candidate_sha256'] == sa.digest(results / 'm3_waveform_candidate_v1/candidate.json')
    frozen = json.loads((source / 'frozen_selections.json').read_text()); seeds = rp['seeds']; steps = [.1, .25, .5, .75, 1.]
    assert seeds == list(range(101, 111))
    out = results / 'm4_random_function_risk_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'protocol.json', dict(seeds=seeds, steps=steps, family='function_private',
        source_summary_sha256=sa.digest(source / 'summary.json'), source_selection_sha256=sa.digest(source / 'frozen_selections.json'),
        source_audit_sha256=sa.digest(source / 'verification.json'), sensitivity_audit_sha256=sa.digest(sensitivity / 'verification.json'),
        risk_protocol_sha256=sa.digest(risk / 'protocol.json'), risk_summary_sha256=sa.digest(risk / 'summary.json'),
        code_sha256=sa.digest(Path(__file__)),
        limits='Post-screen exploratory private-function nomination, not original primary change. All five steps and ten paired M2 source-risk draws. Fixed GPs and validation choices, not full retraining. Report both all32edges and eight M4-addition edges; unchanged parent instabilities not hidden. No new selection by dataset or seed.'))
    cells = {ds:sa.load_cells(ds) for ds in ('XJTU', 'MATR', 'Tongji')}; rows = []; manifest = []
    for fold in frozen['selections']:
        ds = fold['fold'].split(':')[0]; _, _, test = sa.split(cells[ds], fold['fold'])
        gain = fold['choices']['function_private']['selected']['gain']
        for c in test:
            tail = Path('folds') / fold['fold'].replace(':', '__') / 'predictions' / (c.id + '.npz')
            with np.load(source / tail) as z: y = z['y'].copy(); q = z['branch_function_private'].copy()
            assert np.array_equal(y, c.y[sa.K:])
            for seed in seeds:
                path = risk / 'folds' / fold['fold'].replace(':', '__') / str(seed) / (c.id + '.npz')
                with np.load(path) as z:
                    assert np.array_equal(y, z['y'])
                    for g in PARENTS:
                        p = z[g]
                        rows.append(dict(seed=seed, dataset=ds, domain=c.domain, cell_id=c.id, group=g, **error_metrics(y, p)))
                        for step in steps:
                            pred = p + step * gain * (q - p)
                            rows.append(dict(seed=seed, dataset=ds, domain=c.domain, cell_id=c.id, group=g + '4__' + str(step), **error_metrics(y, pred)))
                manifest.append(dict(path=str(path), sha256=sa.digest(path)))
        print(fold['fold'], 'ten paired risk draws', flush=True)
    assert len(rows) == 175200 and len(manifest) == 3650
    table = aggregate(rows, ['seed', 'dataset', 'group']); lookup = {(r['seed'], r['dataset'], r['group']):r for r in table}
    previous = json.loads((risk / 'summary.json').read_text())['table']
    parent_error = max(abs(lookup[r['seed'], r['dataset'], r['group']][k] - r[k]) for r in previous for k in ('mae', 'rmse', 'p95_ae'))
    assert parent_error < 1e-8
    gates = []
    for seed in seeds:
        for step in steps:
            def mapped(g): return g + '__' + str(step) if '4' in g else g
            failures = {a + '<' + b:[dict(dataset=ds, metric=k, delta_pp=lookup[seed, ds, mapped(a)][k] - lookup[seed, ds, mapped(b)][k])
                for ds in cells for k in ('mae', 'rmse', 'p95_ae') if lookup[seed, ds, mapped(a)][k] >= lookup[seed, ds, mapped(b)][k]] for a, b in addition_edges()}
            gates.append(dict(seed=seed, step=step, passed={edge:not rr for edge, rr in failures.items()}, failures=failures))
    counts = {}
    m4_edges = [g + '4<' + g for g in PARENTS]
    for step in steps:
        rr = [r for r in gates if r['step'] == step]
        counts[str(step)] = dict(independent=sum(r['passed']['B4<B'] for r in rr), full=sum(r['passed']['B1234<B123'] for r in rr),
            all_m4_additions=sum(all(r['passed'][e] for e in m4_edges) for r in rr), all32=sum(all(r['passed'].values()) for r in rr))
    dump_csv(out / 'cells.csv', rows); dump_csv(out / 'comparison.csv', table)
    sa.write_json(out / 'summary.json', dict(status='RISK_SCREEN_COMPLETE_AUDIT_PENDING', rows=len(rows), table=table, gates=gates,
        pass_counts=counts, parent_metric_error=parent_error, source_cache_manifest=manifest,
        limits='Fixed-GP risk resampling, not full retraining; independent replay pending. No M4 adoption.'))
    print(counts, flush=True)


if __name__ == '__main__': main()
