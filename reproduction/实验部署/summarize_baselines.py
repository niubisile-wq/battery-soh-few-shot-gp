"""Audit saved predictions, then aggregate seeds and matched cells/domains."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from study import HERE, ROOT, PROTOCOL, PROTOCOL_PATH, digest, write_json, write_csv, preservation
from derive_statistics import paired_summary

OUT=HERE/'baselines'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    preservation()
    expected=json.loads((OUT/'plan.json').read_text())
    rows=[];missing=[];training=[];max_error=0.;complete_hashes={}
    for ds,domain,name in expected['tasks']:
        job=OUT/f'{ds}__{domain}__{name}'
        for seed in PROTOCOL['baseline_seeds']:
            complete=job/f'seed_{seed}'/'complete.json'
            if not complete.exists():missing.append([ds,domain,name,seed]);continue
            done=json.loads(complete.read_text())
            assert done['protocol_sha256']==digest(PROTOCOL_PATH)
            assert done['provenance_sha256']==digest(job/'provenance.json')
            for p,h in done['artifacts'].items():assert digest(p)==h,(p,'hash')
            complete_hashes[str(complete)]=digest(complete)
            rr=list(csv.DictReader((complete.parent/'metrics.csv').open()))
            assert len(rr)==done['rows']==done['cells']*3
            assert len({r['cell_id'] for r in rr})==done['cells']
            for r in rr:
                with np.load(r['prediction_file'],allow_pickle=False) as z:
                    e=np.asarray(z['pred'],dtype=float)-np.asarray(z['y'],dtype=float)
                    assert len(e)==int(r['query_count']) and np.isfinite(e).all()
                    m=dict(mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),p95_ae=float(np.quantile(abs(e),.95)))
                for key,v in m.items():max_error=max(max_error,abs(v-float(r[key])))
                rows.append(dict(dataset=ds,domain=domain,model=name,seed=seed,setting=r['setting'],
                    cell_id=r['cell_id'],selected_mode=r['selected_mode'],**{k:v*100 for k,v in m.items()}))
            record=json.loads((complete.parent/'training.json').read_text())
            assert len(record['boundary'])==done['cells']
            assert all(r['query_labels_masked'] and r['current_query_replay_max_abs']<1e-5 for r in record['boundary'])
            info=record['pretrain']
            training.append(dict(dataset=ds,domain=domain,model=name,seed=seed,
                reused=record['reused_source'] is not None,stop_reason=info['stop_reason'],
                epochs_or_updates=info.get('epochs_run',info.get('outer_updates')),
                best_epoch_or_step=info.get('best_epoch',info.get('best_step')),
                validation_mae=info['best_inner_mae'],selected_mode=record['chosen']['primary']['mode'],
                hard_cap_recent_best=info['stop_reason']=='hard_cap_recent_best'))
    assert max_error<1e-12
    dest=OUT/('partial_summary' if missing else 'summary');dest.mkdir(exist_ok=True)
    write_json(dest/'audit.json',dict(status='PARTIAL' if missing else 'ALL_PREDICTIONS_RECOMPUTED',
        missing=missing,complete_seed_tasks=len(complete_hashes),expected_seed_tasks=expected['seed_tasks'],
        cell_setting_rows=len(rows),metric_max_error_ratio=max_error,complete_hashes=complete_hashes,
        recent_cap_training_records=[r for r in training if r['hard_cap_recent_best']]))
    if rows:write_csv(dest/'cell_seed_metrics.csv',rows)
    if training:write_csv(dest/'training_audit.csv',training)
    if missing:
        print('PARTIAL',len(complete_hashes),'/',expected['seed_tasks'],'seed tasks complete')
        if not args.partial:raise SystemExit('Cannot finalize an incomplete baseline matrix')
        return
    seeds=[]
    for ds in PROTOCOL['datasets']:
        for model in PROTOCOL['baseline_models']:
            for setting in ['inner_selected','source_only','source_bias']:
                for seed in PROTOCOL['baseline_seeds']:
                    rr=[r for r in rows if r['dataset']==ds and r['model']==model and r['setting']==setting and r['seed']==seed]
                    buckets=defaultdict(list)
                    for r in rr:buckets[r['domain']].append(r)
                    seeds.append(dict(dataset=ds,model=model,setting=setting,seed=seed,cells=len(rr),domains=len(buckets),
                        **{k:float(np.mean([np.mean([r[k] for r in group]) for group in buckets.values()])) for k in ['mae','rmse','p95_ae']}))
    table=[]
    for ds in PROTOCOL['datasets']:
        for model in PROTOCOL['baseline_models']:
            for setting in ['inner_selected','source_only','source_bias']:
                rr=[r for r in seeds if r['dataset']==ds and r['model']==model and r['setting']==setting]
                assert len(rr)==3
                values={}
                for k in ['mae','rmse','p95_ae']:
                    values[k+'_mean_pp']=float(np.mean([r[k] for r in rr]))
                    values[k+'_seed_sd_pp']=float(np.std([r[k] for r in rr],ddof=1))
                table.append(dict(dataset=ds,model=model,setting=setting,seeds=3,cells=rr[0]['cells'],domains=rr[0]['domains'],**values))
    write_csv(dest/'seed_dataset_metrics.csv',seeds);write_csv(dest/'comparison.csv',table)
    frozen=list(csv.DictReader((HERE/'statistics/cell_metrics.csv').open()))
    final={(r['dataset'],r['domain'],Path(r['cell_id']).name):r for r in frozen if r['group']=='B1234'}
    paired=[]
    for ds in PROTOCOL['datasets']:
        for model in PROTOCOL['baseline_models']:
            buckets=defaultdict(list)
            for r in rows:
                if r['dataset']==ds and r['model']==model and r['setting']=='inner_selected':
                    buckets[(r['dataset'],r['domain'],Path(r['cell_id']).name)].append(r)
            assert set(buckets)=={key for key in final if key[0]==ds}
            for k in ['mae','rmse','p95_ae']:
                keys=sorted(buckets)
                diff=[float(final[key][k])-float(np.mean([r[k] for r in buckets[key]])) for key in keys]
                paired.append(dict(dataset=ds,comparison='B1234-'+model,metric=k,
                    baseline_estimator='mean_of_per_cell_metrics_over_three_seeds',**paired_summary(diff,[key[1] for key in keys])))
    write_csv(dest/'paired_vs_final.csv',paired)
    print('BASELINES_RECOMPUTED',len(complete_hashes),'seed tasks;',len(rows),'cell-setting rows;',len(paired),'paired comparisons')
    for r in table:
        if r['setting']=='inner_selected':print(r)


if __name__=='__main__':main()
