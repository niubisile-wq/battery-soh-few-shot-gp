"""Recompute cell metrics and keep per-dataset completion explicit."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from data import ROOT,OUT,load_cells,write_json,metrics
from screen import macro


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-name',default='screen_v1');ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    root=OUT/args.out_name;rows=[];jobs=[];metric_error=0.;points=0
    for p in sorted((root/'jobs').glob('*/complete.json')):
        jobs.append(json.loads(p.read_text()))
        rs=json.loads((p.parent/'cell_results.json').read_text())
        if args.verify:
            for r in rs:
                with np.load(ROOT/r['prediction_file']) as z:
                    m=metrics(z['y'],z['pred']);points+=len(z['y'])
                for key in m:
                    if m[key] is not None:metric_error=max(metric_error,abs(m[key]-r[key]))
        rows.extend(rs)
    groups=defaultdict(list)
    for r in rows:groups[(r['candidate'],r['dataset'],'LODO' if r['fold'].startswith('LODO:') else 'within')].append(r)
    summary=[]
    expected={ds:{c.id for c in load_cells(ds)} for ds in sorted({r['dataset'] for r in rows})}
    for (candidate,ds,track),rs in sorted(groups.items()):
        ids=[r['cell_id'] for r in rs]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate candidate/dataset cell scores')
        summary.append({'candidate':candidate,'dataset':ds,'track':track,'cells':len(rs),
                        'expected_cells':len(expected[ds]),'complete':set(ids)==expected[ds],
                        **{k:macro(rs,k) for k in ['mae','rmse','p95_ae','max_ae','bias','low_soh_mae']}})
    summary.sort(key=lambda r:(r['track'],r['dataset'],r['mae']))
    if rows:
        with (root/'cell_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (root/'summary.csv').open('w',newline='') as f:
        if summary:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines=['# First-module development screen','',
           'Exploratory development results. Ordinary enhancements are controls, not new-module claims. Missing cells prevent a full-dataset ranking.','',
           '| Dataset | Track | Candidate | Coverage | MAE (pp) | RMSE (pp) | P95 AE (pp) |',
           '|---|---|---|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["dataset"]} | {r["track"]} | {r["candidate"]} | {r["cells"]}/{r["expected_cells"]} | {100*r["mae"]:.4f} | {100*r["rmse"]:.4f} | {100*r["p95_ae"]:.4f} |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    result={'completed_jobs':len(jobs),'cell_result_rows':len(rows),'prediction_points_verified':points,
            'maximum_metric_recomputation_error':metric_error if args.verify else None,'summary':summary}
    write_json(root/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
