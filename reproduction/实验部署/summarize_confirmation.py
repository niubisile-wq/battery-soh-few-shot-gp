"""Recompute every metric and independently replay frozen cohort predictions."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
from dataclasses import replace
import csv
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from study import HERE,BUNDLE,digest,write_json,write_csv,preservation
from derive_statistics import paired_summary

sys.path.insert(0,str(BUNDLE/'source_snapshot/m4'))
import random_function_module
from data import Cell

OUT=HERE/'external/confirmation'
METRICS=['mae_pp','rmse_pp','p95_ae_pp']


def replay(fold):
    request=json.loads((OUT/'request.json').read_text());lock=json.loads((HERE/'external/exposure_and_source_lock.json').read_text())
    rows=[]
    with threadpool_limits(limits=1):
        models={}
        for key in ['B4','B14','B124','B1234']:
            path=BUNDLE/'folds'/fold.replace(':','__')/key/'adapter.joblib'
            assert digest(path)==lock['source_models'][str(path)];models[key]=joblib.load(path)
        for spec in request['cells']:
            assert digest(spec['cache'])==spec['cache_sha256']
            with np.load(spec['cache'],allow_pickle=False) as z:
                x=z['x'].astype('float32');q=z['capacity_Ah'].astype('float32');cycle=z['cycle_number'].astype('float32')
            y=q/q[0];visible=np.full_like(y,np.nan);visible[:10]=y[:10]
            cell=Cell(spec['cell_id'],'CALCE_CONFIRMATION',spec['domain'],x,visible,cycle,float(q[0]),spec['nominal_capacity_Ah'],spec['cache'])
            for group,key in [('B','B4'),('B1','B14'),('B12','B124'),('B123','B1234'),('B1234','B1234')]:
                adapter=models[key]
                if group=='B1234':p=adapter.predict(cell)
                elif adapter.parent_mode is None:p=adapter.parent.predict(cell)
                else:p=adapter.parent.predict(cell,adapter.parent_mode)
                path=OUT/'folds'/fold.replace(':','__')/'predictions'/cell.id/(group+'.npz')
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['y'],y[10:]);np.testing.assert_array_equal(z['cycle'],cycle[10:]);np.testing.assert_array_equal(z['support_y'],y[:10])
                    delta=float(np.max(abs(p-z['pred'])))
                assert delta<1e-9,(fold,cell.id,group,delta)
                rows.append(dict(source_fold=fold,cell_id=cell.id,group=group,replayed_query_points=len(p),max_replay_abs=delta))
    print('CONFIRMATION_REPLAY',fold,len(rows),flush=True);return rows


def main():
    preservation();request=json.loads((OUT/'request.json').read_text())
    done=json.loads((OUT/'inference_complete.json').read_text());assert len(done['folds'])==6
    rows=[];bounds=[];hashes={};maximum=0.
    for fold in request['source_folds']:
        folder=OUT/'folds'/fold.replace(':','__');complete=json.loads((folder/'complete.json').read_text())
        assert complete['status']=='FROZEN_INFERENCE_COMPLETE' and complete['request_sha256']==digest(OUT/'request.json')
        for p,h in complete['artifacts'].items():assert digest(p)==h;hashes[p]=h
        for r in csv.DictReader((folder/'metrics.csv').open()):
            with np.load(r['prediction_file'],allow_pickle=False) as z:
                e=(z['pred'].astype(float)-z['y'].astype(float))*100;low=z['y'].astype(float)<.9
            a=abs(e);m=dict(mae_pp=float(a.mean()),rmse_pp=float(np.sqrt(np.mean(e*e))),p95_ae_pp=float(np.quantile(a,.95)))
            assert len(e)==int(r['query'])
            for k,v in m.items():maximum=max(maximum,abs(v-float(r[k])))
            rows.append({**r,**m,'query':len(e),'low_soh_query':int(low.sum()),'low_soh_mae_pp':float(a[low].mean()) if low.any() else None})
        b=json.loads((folder/'boundary.json').read_text())
        assert len(b)==len(request['cells'])*5
        assert all(r['query_labels_hidden'] and r['source_fitting_disabled'] and r['altered_query_labels_max_abs']<1e-8 and r['current_query_max_abs']<1e-7 for r in b)
        bounds.extend(b)
    assert maximum<1e-10 and len(rows)==210
    assert len({(r['source_fold'],r['cell_id'],r['group']) for r in rows})==210
    replay_rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(replay,fold) for fold in request['source_folds']]):replay_rows.extend(f.result())
    dest=OUT/'summary';dest.mkdir(exist_ok=True)
    write_csv(dest/'cell_fold_metrics.csv',rows);write_csv(dest/'independent_replay.csv',replay_rows);write_csv(dest/'boundary.csv',bounds)
    domains=sorted({r['domain'] for r in rows});domain_fold=[];folds=[];cells=[];domain_average=[];overall=[]
    for fold in request['source_folds']:
        for group in request['groups']:
            for domain in domains:
                rr=[r for r in rows if (r['source_fold'],r['group'],r['domain'])==(fold,group,domain)]
                domain_fold.append(dict(source_fold=fold,group=group,domain=domain,cells=len(rr),
                    **{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
            rr=[r for r in domain_fold if (r['source_fold'],r['group'])==(fold,group)]
            folds.append(dict(source_fold=fold,group=group,protocol_groups=len(rr),cells=len(request['cells']),
                **{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
    for spec in request['cells']:
        for group in request['groups']:
            rr=[r for r in rows if (r['cell_id'],r['group'])==(spec['cell_id'],group)]
            cells.append(dict(cell_id=spec['cell_id'],domain=spec['domain'],group=group,source_folds=len(rr),query=spec['query'],
                **{k:float(np.mean([r[k] for r in rr])) for k in METRICS},
                **{k+'_'+suffix:float(fn([r[k] for r in rr])) for k in METRICS for suffix,fn in [('source_min',np.min),('source_max',np.max)]}))
    for group in request['groups']:
        rr=[r for r in folds if r['group']==group]
        overall.append(dict(group=group,source_folds=6,protocol_groups=len(domains),cells=len(request['cells']),
            **{k:float(np.mean([r[k] for r in rr])) for k in METRICS},
            **{k+'_'+suffix:float(fn([r[k] for r in rr])) for k in METRICS for suffix,fn in [('source_min',np.min),('source_max',np.max)]}))
        for domain in domains:
            rr=[r for r in domain_fold if (r['group'],r['domain'])==(group,domain)]
            domain_average.append(dict(group=group,domain=domain,cells=rr[0]['cells'],source_folds=6,
                **{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
    by={(r['cell_id'],r['group']):r for r in cells};by_fold={(r['source_fold'],r['cell_id'],r['group']):r for r in rows}
    paired=[]
    for comparison in ['B1234-B','B1-B','B12-B1','B123-B12','B1234-B123']:
        a,b=comparison.split('-')
        for metric in METRICS:
            diffs=[by[s['cell_id'],a][metric]-by[s['cell_id'],b][metric] for s in request['cells']]
            raw_diffs=[by_fold[f,s['cell_id'],a][metric]-by_fold[f,s['cell_id'],b][metric] for f in request['source_folds'] for s in request['cells']]
            paired.append(dict(comparison=comparison,metric=metric,
                **paired_summary(diffs,[s['domain'] for s in request['cells']]),
                improved_cell_source_pairs=int((np.asarray(raw_diffs)<-1e-10).sum()),
                worse_cell_source_pairs=int((np.asarray(raw_diffs)>1e-10).sum()),
                estimand='Mean of per-cell errors across6 frozen source folds; cells stratified in4 protocol groups; descriptive tiny-cohort CI, not independent-source CI'))
    for filename,rr in [('domain_fold_metrics.csv',domain_fold),('source_fold_metrics.csv',folds),('cell_mean_across_sources.csv',cells),
                        ('domain_mean_across_sources.csv',domain_average),('cohort_summary.csv',overall),('paired_intervals.csv',paired)]:write_csv(dest/filename,rr)
    audit=dict(status='FROZEN_CONFIRMATION_VERIFIED',cells=len(request['cells']),source_folds=6,groups=5,cell_group_source_rows=len(rows),
        physical_cell_count=7,unique_measured_queries=sum(s['query'] for s in request['cells']),no_core_training=True,no_target_selection=True,
        no_ensembling=True,metric_recompute_max_error_pp=maximum,max_prediction_replay_abs=max(r['max_replay_abs'] for r in replay_rows),
        replayed_predictions=len(replay_rows),max_query_label_perturbation_abs=max(r['altered_query_labels_max_abs'] for r in bounds),
        max_current_query_abs=max(r['current_query_max_abs'] for r in bounds),
        request_sha256=digest(OUT/'request.json'),input_artifact_sha256=hashes,code_sha256=digest(Path(__file__)),
        output_sha256={p.name:digest(p) for p in dest.glob('*.csv')},
        limits='7 distinct cells,4 protocol groups, same CALCE laboratory data source previously used. Six source folds overlap, not6 independent replicates. First10 eligible measured pairs may be RPT; rate-specific operational capacity; no population-wide claim.')
    write_json(dest/'audit.json',audit)
    for r in overall:print('COHORT',r['group'],*[round(r[k],6) for k in METRICS])
    for s in request['cells']:print('CELL',s['cell_id'],*[round(by[s['cell_id'],g]['mae_pp'],6) for g in request['groups']])
    print('CONFIRMATION_FULLY_VERIFIED',len(rows),'cell-group-source rows',flush=True)


if __name__=='__main__':main()
