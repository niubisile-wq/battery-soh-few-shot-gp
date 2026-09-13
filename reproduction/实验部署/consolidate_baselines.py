"""Join evidence blocks without concealing their different selection policies."""
from collections import defaultdict
import csv
import numpy as np
from study import HERE,ROOT,WORK,digest,write_csv,write_json


def main():
    old=WORK/'论文修订版_实验问题修复/证据/基线/multidataset_classical_v1'
    raw=list(csv.DictReader((old/'cell_results.csv').open()));summary=list(csv.DictReader((old/'summary.csv').open()))
    maximum=0.;rows=[]
    for s in summary:
        ds=s['dataset'];method=s['method'];rr=[r for r in raw if (r['dataset'],r['method'])==(ds,method)]
        assert {r['setting'] for r in rr}=={s['setting']}
        seeds=sorted({r['seed'] for r in rr});assert len(seeds)==int(s['n_seeds']);m={}
        for key in ['mae','rmse','p95_ae']:
            sv=[]
            for seed in seeds:
                domain=defaultdict(list);zz=[r for r in rr if r['seed']==seed]
                assert len({r['cell_id'] for r in zz})==len(zz)==dict(XJTU=55,MATR=180,Tongji=130)[ds]
                for r in zz:domain[r['domain']].append(float(r[key]))
                sv.append(np.mean([np.mean(v) for v in domain.values()]))
            maximum=max(maximum,abs(float(np.mean(sv))-float(s[key])))
            m[key+'_mean_pp']=float(np.mean(sv)*100);m[key+'_seed_sd_pp']=float(np.std(sv,ddof=1)*100) if len(sv)>1 else None
        rows.append(dict(dataset=ds,method=method,setting=s['setting'],seeds=len(seeds),
            evidence='existing_cell_metrics_reaggregated_not_checkpoint_replayed',
            selection_policy='one_XJTU_frozen_config_and_setting_per_method',**m))
    assert maximum<1e-12
    new=list(csv.DictReader((HERE/'baselines/summary/comparison.csv').open()))
    for r in new:
        if r['setting']=='inner_selected':
            rows.append(dict(dataset=r['dataset'],method=r['model'],setting=r['setting'],seeds=int(r['seeds']),
                evidence='new_study_checkpoints_all_replayed',selection_policy='fold_local_source_validation',
                **{k:float(r[k]) for k in r if k.endswith('_pp')}))
    frozen=list(csv.DictReader((HERE/'statistics/full_ablation_16_groups.csv').open()))
    for r in frozen:
        if r['group']=='B1234':
            rows.append(dict(dataset=r['dataset'],method='B1234_frozen',setting='frozen_complete_model',seeds=1,
                evidence='frozen_development_predictions_recomputed',selection_policy='historically_exposed_development_selection',
                **{k+'_mean_pp':float(r[k]) for k in ['mae','rmse','p95_ae']}))
    dest=HERE/'baselines/summary';write_csv(dest/'combined_12_baselines_plus_final.csv',rows)
    old19=list(csv.DictReader((ROOT/'开发基线选择依据/results/fair_selection_v1/comparison.csv').open()))
    methods=sorted({r['method'] for r in old19 if r['track']=='strict'});assert len(methods)==19
    classical={r['method'] for r in summary};deep={'PatchTSTLite','TCN','MAML_MLP'}
    coverage=[dict(method=m,XJTU_original='complete',
        XJTU_this_supplement='checkpoint_verified_3_seeds' if m in deep else 'existing',
        MATR='checkpoint_verified_3_seeds' if m in deep else 'existing_classical' if m in classical else 'not_in_this_supplement',
        Tongji='checkpoint_verified_3_seeds' if m in deep else 'existing_classical' if m in classical else 'not_in_this_supplement') for m in methods]
    write_csv(dest/'method_coverage.csv',coverage)
    write_json(dest/'legacy_aggregation_audit.json',dict(status='EXISTING_CELL_METRICS_REAGGREGATED',rows=len(raw),
        summary_rows=len(summary),max_error_ratio=maximum,code_sha256=digest(__file__),
        source_sha256={str(p):digest(p) for p in [old/'cell_results.csv',old/'summary.csv',old/'manifest.json',old.parent/'run_multidataset_classical.py']},
        limitations='Existing archive has per-cell metrics but not archived per-point predictions and model checkpoints in this evidence directory; do not claim the same checkpoint-replay audit as the new three-family study. Its global frozen configuration policy differs from new fold-local source-validation selection.',
        output_sha256={p.name:digest(p) for p in [dest/'combined_12_baselines_plus_final.csv',dest/'method_coverage.csv']}))
    print('COMBINED',len(rows),'rows; coverage',len(coverage),'methods; old aggregate max difference',maximum)


if __name__=='__main__':main()
