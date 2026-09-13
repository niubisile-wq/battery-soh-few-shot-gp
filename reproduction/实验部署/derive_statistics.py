"""Recompute descriptive evidence from every final frozen prediction, without fit."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from study import HERE, ROOT, BUNDLE, PROTOCOL, PROTOCOL_PATH, digest, write_json, write_csv, preservation

OUT = HERE/'statistics'
METRICS = ('mae','rmse','p95_ae')
GROUPS = ('B','B1','B2','B3','B4','B12','B13','B14','B23','B24','B34','B123','B124','B134','B234','B1234')


def metrics(y,p):
    e=(np.asarray(p,dtype=float)-np.asarray(y,dtype=float))*100
    if not len(e): return {k:None for k in METRICS}
    if not np.isfinite(e).all(): raise ValueError('Nonfinite frozen prediction')
    return dict(mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),p95_ae=float(np.quantile(abs(e),.95)))


def equal_domain_mean(rows, key):
    groups=defaultdict(list)
    for r in rows:
        if r[key] is not None: groups[r['domain']].append(float(r[key]))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def paired_summary(values, domains, repetitions=None):
    reps=PROTOCOL['statistics']['bootstrap_replicates'] if repetitions is None else repetitions
    x=np.asarray(values,dtype=float);ds=np.asarray(domains)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid paired values')
    groups=[x[ds==d] for d in sorted(set(domains))]
    rng=np.random.default_rng(PROTOCOL['statistics']['bootstrap_seed'])
    cell=np.zeros(reps)
    for g in groups:
        cell+=g[rng.integers(0,len(g),size=(reps,len(g)))].mean(axis=1)/len(groups)
    means=np.asarray([g.mean() for g in groups])
    domain=means[rng.integers(0,len(groups),size=(reps,len(groups)))].mean(axis=1)
    return dict(delta_pp=float(means.mean()),cells=len(x),domains=len(groups),
        improved_cells=int((x < -1e-10).sum()),worse_cells=int((x > 1e-10).sum()),tied_cells=int((abs(x)<=1e-10).sum()),
        improved_domains=int((means < -1e-10).sum()),worse_domains=int((means > 1e-10).sum()),
        cell_ci_low_pp=float(np.quantile(cell,.025)),cell_ci_high_pp=float(np.quantile(cell,.975)),
        domain_ci_low_pp=float(np.quantile(domain,.025)),domain_ci_high_pp=float(np.quantile(domain,.975)),
        median_cell_delta_pp=float(np.median(x)),bootstrap_reps=reps)


def main():
    preservation()
    OUT.mkdir(exist_ok=True)
    candidate=json.loads((BUNDLE/'candidate.json').read_text())
    records=[];source_hashes={};point_errors=defaultdict(list)
    paths=sorted((BUNDLE/'folds').glob('*/predictions/*/*.npz'))
    assert len(paths)==365
    expected={str(BUNDLE/r['path']):r['sha256'] for r in candidate['predictions']}
    for path in paths:
        sha=digest(path);assert sha==expected[str(path)]
        source_hashes[str(path)]=sha
        ds,domain=path.parts[-4].split('__',1)
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)==set(GROUPS)|{'y'}
            y=np.asarray(z['y'],dtype=float);low=y<.90
            for group in GROUPS:
                values=metrics(y,z[group]);lm=metrics(y[low],z[group][low])
                records.append(dict(dataset=ds,domain=domain,cell_id=path.stem,group=group,n=len(y),low_n=int(low.sum()),
                                    **values,**{'low_'+k:v for k,v in lm.items()}))
                if group in ('B','B1234'):
                    point_errors[ds,group].append((z[group][low].astype(float)-y[low].astype(float))*100)
    write_csv(OUT/'cell_metrics.csv',records)
    aggregate=[]
    for ds in PROTOCOL['datasets']:
        for group in GROUPS:
            rr=[r for r in records if r['dataset']==ds and r['group']==group]
            aggregate.append(dict(dataset=ds,group=group,cells=len(rr),domains=len({r['domain'] for r in rr}),
                points=sum(r['n'] for r in rr),low_cells=sum(r['low_n']>0 for r in rr),low_points=sum(r['low_n'] for r in rr),
                low_domains=len({r['domain'] for r in rr if r['low_n']>0}),
                **{m:equal_domain_mean(rr,m) for m in METRICS+tuple('low_'+m for m in METRICS)}))
    source=list(csv.DictReader((BUNDLE/'comparison.csv').open()))
    lookup={(r['dataset'],r['group']):r for r in aggregate}
    err=max(abs(float(r[m])-lookup[r['dataset'],r['group']][m]) for r in source for m in METRICS)
    assert err<1e-8,err
    write_csv(OUT/'full_ablation_16_groups.csv',aggregate)
    paired=[]
    for ds in PROTOCOL['datasets']:
        by={(r['domain'],r['cell_id'],r['group']):r for r in records if r['dataset']==ds}
        cells=sorted({(d,c) for d,c,g in by})
        for comparison in PROTOCOL['statistics']['comparisons']:
            a,b=comparison.split('-')
            for m in METRICS+('low_mae','low_rmse','low_p95_ae'):
                valid=[(d,c) for d,c in cells if by[d,c,a][m] is not None and by[d,c,b][m] is not None]
                pp=paired_summary([by[d,c,a][m]-by[d,c,b][m] for d,c in valid],[d for d,c in valid])
                delta=lookup[ds,a][m]-lookup[ds,b][m]
                assert abs(delta-pp['delta_pp'])<1e-10
                paired.append(dict(dataset=ds,comparison=comparison,metric=m,**pp,excluded_no_low_soh_cells=len(cells)-len(valid)))
    write_csv(OUT/'paired_intervals.csv',paired)
    write_csv(OUT/'cell_win_loss.csv',[r for r in paired if r['comparison']=='B1234-B' and r['metric'] in METRICS])
    write_csv(OUT/'low_soh_domain_weighted.csv',[r for r in aggregate if r['group'] in ['B','B1234']])
    points=[]
    for (ds,group),chunks in sorted(point_errors.items()):
        e=np.concatenate(chunks)
        points.append(dict(dataset=ds,group=group,points=len(e),mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),
                           p95_ae=float(np.quantile(abs(e),.95)),aggregation='pooled_query_points_not_main_estimand'))
    write_csv(OUT/'low_soh_pooled_points_secondary.csv',points)
    controls={
        'early_B12':ROOT/'模块研发/results/frozen_v3_evaluation_v2/comparison.csv',
        'final_M4':ROOT/'模块研发/results/m4_random_function_private_controls_v1/comparison.csv'}
    for stage,path in controls.items():
        rr=list(csv.DictReader(path.open()))
        write_csv(OUT/f'controls_{stage}.csv',[dict(stage=stage,**r) for r in rr])
    for path,sha in source_hashes.items():assert digest(path)==sha
    preservation()
    manifest=dict(status='RECOMPUTED_FROZEN_STATISTICS',training_performed=False,
        protocol_sha256=digest(PROTOCOL_PATH),code_sha256=digest(Path(__file__)),prediction_file_count=len(paths),
        prediction_sha256=source_hashes,comparison_max_error_pp=err,cell_rows=len(records),paired_rows=len(paired),
        controls_sha256={k:digest(p) for k,p in controls.items()},limitations=PROTOCOL['statistics']['limits'],
        outputs={p.name:digest(p) for p in OUT.glob('*.csv')})
    write_json(OUT/'manifest.json',manifest)
    print('RECOMPUTED',len(records),'cell rows;',len(paired),'paired intervals; max metric difference',err)


if __name__=='__main__':main()
