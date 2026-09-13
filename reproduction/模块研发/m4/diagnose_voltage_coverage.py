"""Label-free descriptive audit of registered voltage grid coverage."""
import json
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa,HERE
from evaluate import dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(HERE.parent/'results/m4_voltage_coverage_v1'));args=ap.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=False);rows=[]
    for dataset in ('XJTU','MATR','Tongji'):
        for c in sa.load_cells(dataset):
            voltage=np.asarray(c.x[:,0],float)
            lo=float(voltage[:sa.K].min(1).mean());hi=float(voltage[:sa.K].max(1).mean())
            assert hi>lo+1e-8
            grid=np.linspace(lo,hi,64)
            # Monotone envelope begins at first voltage, ends at maximum observed.
            coverage=((grid[None,:]>=voltage[:,0,None])&(grid[None,:]<=voltage.max(1)[:,None])).mean(1)
            query=coverage[sa.K:]
            shifts=np.maximum(abs(voltage[sa.K:].min(1)-lo),abs(voltage[sa.K:].max(1)-hi))
            rows.append(dict(dataset=dataset,domain=c.domain,cell_id=c.id,query_rows=len(query),
                max_endpoint_shift_V=float(shifts.max()),query_count_endpoint_shift_over_1mV=int(np.sum(shifts>.001)),
                reference_lo=lo,reference_hi=hi,support_mean_coverage=float(coverage[:sa.K].mean()),
                query_mean_coverage=float(query.mean()),query_min_coverage=float(query.min()),
                query_fraction_below_half=float(np.mean(query<.5)),
                query_fraction_full_coverage=float(np.mean(query==1.))))
        print(dataset,'input-only coverage checked',flush=True)
    aggregates=[]
    for dataset in ('XJTU','MATR','Tongji'):
        rr=[r for r in rows if r['dataset']==dataset];domains={r['domain'] for r in rr}
        aggregates.append(dict(dataset=dataset,cells=len(rr),domains=len(domains),
            max_endpoint_shift_V=max(r['max_endpoint_shift_V'] for r in rr),
            query_weighted_fraction_endpoint_shift_over_1mV=sum(r['query_count_endpoint_shift_over_1mV'] for r in rr)/sum(r['query_rows'] for r in rr),
            domain_equal_mean_coverage=float(np.mean([np.mean([r['query_mean_coverage'] for r in rr if r['domain']==d]) for d in domains])),
            domain_equal_fraction_below_half=float(np.mean([np.mean([r['query_fraction_below_half'] for r in rr if r['domain']==d]) for d in domains])),
            cells_with_any_below_half=sum(r['query_fraction_below_half']>0 for r in rr)))
    assert len(rows)==365
    dump_csv(out/'cells.csv',rows)
    sa.write_json(out/'summary.json',dict(status='LABEL_FREE_COVERAGE_DIAGNOSIS_COMPLETE',table=aggregates,
        code_sha256=sa.digest(Path(__file__)),limits='Reads voltage signals only,not query SOH. Descriptive coverage does not measure model accuracy or authorize per-test-cell filtering.'))
    print(json.dumps(aggregates),flush=True)


if __name__=='__main__':main()
