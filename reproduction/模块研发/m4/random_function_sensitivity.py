"""All registered families/steps; no new fits or per-dataset selection."""
import json
from pathlib import Path
import numpy as np
from source_oof import sa, HERE
from score import PARENTS, addition_edges
from evaluate import aggregate, error_metrics, dump_csv


def name(group, family, step):
    return group + '__' + family + '__' + str(step) if '4' in group else group


def main():
    results = HERE.parent / 'results'; source = results / 'm4_random_function_screen_v1'
    audit = json.loads((source / 'verification.json').read_text())
    assert audit['status'] == 'PASS' and audit['summary_sha256'] == sa.digest(source / 'summary.json')
    protocol = json.loads((HERE / 'random_function_sensitivity_protocol.json').read_text())
    out = results / 'm4_random_function_sensitivity_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'request.json', dict(protocol=protocol, source_sha256=sa.digest(source / 'summary.json'),
        audit_sha256=sa.digest(source / 'verification.json'), frozen_sha256=sa.digest(source / 'frozen_selections.json'),
        code_sha256=sa.digest(Path(__file__)), protocol_sha256=sa.digest(HERE / 'random_function_sensitivity_protocol.json')))
    rows = []
    for ds in ('XJTU', 'MATR', 'Tongji'):
        for c in sa.load_cells(ds):
            path = source / 'folds' / (ds + '__' + c.domain) / 'predictions' / (c.id + '.npz')
            with np.load(path) as z:
                y = z['y']; assert np.array_equal(y, c.y[sa.K:])
                for g in PARENTS:
                    rows.append(dict(dataset=ds, domain=c.domain, cell_id=c.id, group=g, **error_metrics(y, z[g])))
                    for f in protocol['families']:
                        child = g + '4' + ('_' + f if f != 'function_uniform' else '')
                        for step in protocol['global_steps']:
                            p = z[g] + step * (z[child] - z[g])
                            rows.append(dict(dataset=ds, domain=c.domain, cell_id=c.id, group=name(g + '4', f, step), **error_metrics(y, p)))
        print(ds, 'all five families and five steps', flush=True)
    assert len(rows) == 75920
    table = aggregate(rows, ['dataset', 'group']); lookup = {(r['dataset'], r['group']):r for r in table}
    failures = {}; gates = {}
    for f in protocol['families']:
        for step in protocol['global_steps']:
            tag = f + '__' + str(step)
            failures[tag] = {a + '<' + b:[dict(dataset=ds, metric=k, delta_pp=lookup[ds, name(a, f, step)][k] - lookup[ds, name(b, f, step)][k])
                for ds in ('XJTU', 'MATR', 'Tongji') for k in ('mae', 'rmse', 'p95_ae')
                if lookup[ds, name(a, f, step)][k] >= lookup[ds, name(b, f, step)][k]] for a, b in addition_edges()}
            gates[tag] = {edge:not rr for edge, rr in failures[tag].items()}
    dump_csv(out / 'cells.csv', rows); dump_csv(out / 'comparison.csv', table)
    passing = [tag for tag, gg in gates.items() if all(gg.values())]
    sa.write_json(out / 'summary.json', dict(status='SENSITIVITY_COMPLETE_NOT_ADOPTED', rows=len(rows), table=table,
        failures=failures, gates=gates, passing_candidates=passing))
    print('passing', passing, flush=True)
    print('edge counts', {tag:sum(gg.values()) for tag, gg in gates.items()}, flush=True)


if __name__ == '__main__': main()
