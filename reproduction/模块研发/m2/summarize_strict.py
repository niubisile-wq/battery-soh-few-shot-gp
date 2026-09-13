"""Independently audit stored predictions and build the four-group report."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'m1'))
from data import ROOT,K,load_cells,split,source_indices,digest,write_json
import strict_ablation as sa

KEYS=['mae','rmse','p95_ae','low_soh_mae']
DATASETS=sa.PROTOCOL['datasets']


def recalculate(y,p):
    # Compare the stored numeric values to the same float64 SOH threshold as
    # the frozen metric protocol (float32 comparison rounds 0.90 differently).
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    assert y.shape==p.shape and y.size and np.isfinite(p).all() and np.isfinite(y).all()
    e=p.astype(float)-y.astype(float);a=np.abs(e);lo=y<.90
    return {'mae':float(np.sum(a)/len(a)),'rmse':float(np.sqrt(np.dot(e,e)/len(e))),
            'p95_ae':float(np.percentile(a,95)), 'max_ae':float(a.max()),
            'bias':float(e.mean()),'n':len(y),'low_soh_n':int(lo.sum()),
            'low_soh_mae':float(a[lo].mean()) if lo.any() else None,
            'low_soh_bias':float(e[lo].mean()) if lo.any() else None}


def equal_domain(rows,key):
    groups=defaultdict(list)
    for r in rows:
        if r.get(key) is not None: groups[r['domain']].append(r[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def paired_stats(differences,domains,seed=9108):
    x=np.asarray(differences,dtype=float);ds=np.asarray(domains)
    grouped=[x[ds==d] for d in sorted(set(domains))]
    rng=np.random.default_rng(seed);nboot=5000
    cell_boot=np.zeros(nboot)
    for g in grouped:
        cell_boot+=g[rng.integers(0,len(g),size=(nboot,len(g)))].mean(1)/len(grouped)
    means=np.asarray([g.mean() for g in grouped])
    domain_boot=means[rng.integers(0,len(means),size=(nboot,len(means)))].mean(1)
    return {'delta_pp':float(means.mean()*100),
            'ci95_cell_stratified_pp':(np.quantile(cell_boot,[.025,.975])*100).tolist(),
            'ci95_domain_bootstrap_pp':(np.quantile(domain_boot,[.025,.975])*100).tolist(),
            'improved_cells':int(np.sum(x < -1e-12)), 'worse_cells':int(np.sum(x > 1e-12)),
            'tied_cells':int(np.sum(abs(x)<=1e-12)), 'cells':len(x),
            'improved_domains':int(np.sum(means < -1e-12)),'domains':len(means)}


def audit(root):
    cells=[c for ds in DATASETS for c in load_cells(ds)];byid={c.id:c for c in cells}
    folds=sorted({f'{c.dataset}:{c.domain}' for c in cells})
    rows=[];points=0;max_metric_error=0.;runtime=[];source_counts=[];selection_counts=[]
    max_prediction_replay_error=0.
    seen=set();snapshot=sa.signature();boundary=[]
    request=json.loads((root/'request.json').read_text())
    assert request['status']=='COMPLETE' and set(request['folds'])==set(folds)
    assert request['signature']==snapshot
    for fold in folds:
        job=root/'folds'/fold.replace(':','__')
        done=json.loads((job/'complete.json').read_text())
        assert done['status']=='COMPLETE' and done['signature']==snapshot
        runtime.append(done['elapsed_s']);boundary.append(done['inference_audit'])
        assert max(done['inference_audit'].values())<1e-8
        tr,va,te=split(cells,fold);idx=source_indices(tr)
        allowed={(c.id,int(j)) for c in tr for j in idx[c.id]}
        tids={c.id for c in te};vids={c.id for c in va};sids={c.id for c in tr}
        assert not tids&vids and not tids&sids and not vids&sids
        prov=json.loads((job/'provenance.json').read_text())
        assert {tuple(k) for k in prov['source_keys']}==allowed and len(allowed)<=1000
        assert set(prov['validation_cells'])==vids and set(prov['test_cells'])==tids
        for cid,sha in prov['data_sha256'].items(): assert digest(ROOT/byid[cid].path)==sha
        for group,fp in prov['frozen'].items():
            oldjob=ROOT/fp['job']
            assert digest(oldjob/'model.joblib')==fp['model_sha256']
            assert digest(oldjob/'tuning.json')==fp['tuning_sha256']
            assert fp['refit_max_prediction_error']<1e-8
        source_counts.append(len(allowed))
        # Cache only model outputs; truth arrays never enter this cache.
        prediction_cache={}
        for group in ['Base','Base+M1','Physical_control']:
            fr=prov['frozen'][group]
            model=joblib.load(ROOT/fr['job']/'model.joblib')
            prediction_cache[group]={c.id:sa.predict_components(model,c,fr['mode']) for c in va+te}
        adapter_cache={}
        for seed in sa.PROTOCOL['seeds']:
            for group in ['Base+M2','Base+M1+M2']:
                path=job/f'seed_{seed}'/group
                selection=json.loads((path/'selection.json').read_text())
                assert set(selection['selection_cell_ids'])==vids
                assert not set(selection['selection_cell_ids'])&tids
                cf=json.loads((path/'crossfit_audit.json').read_text())
                keys=[tuple(k) for k in cf['keys']]
                assert len(keys)==len(set(keys))==cf['n']
                assert set(keys)=={k for k in allowed if k[1]>=K}
                fit_union=set();hold_union=set()
                for h in cf['crossfit']:
                    fit=set(h['fit_cells']);held=set(h['held_cells'])
                    assert not fit&held and fit|held==sids and not (fit|held)&(vids|tids)
                    fitkeys={tuple(k) for k in h['fit_keys']};heldkeys={tuple(k) for k in h['calibration_keys']}
                    assert fitkeys=={k for k in allowed if k[0] in fit}
                    assert heldkeys=={k for k in allowed if k[0] in held and k[1]>=K}
                    fit_union |= fitkeys;hold_union |= heldkeys
                assert fit_union<=allowed and hold_union<=allowed
                model=joblib.load(path/'adapter.joblib')
                assert model['selection']==selection
                with np.load(path/'calibration_oof.npz') as z:
                    truth=np.asarray([byid[cid].y[j] for cid,j in keys])
                    assert np.array_equal(z['y'],truth)
                    fitted=sa.IsotonicRegression(increasing=True,out_of_bounds='clip').fit(z['pred'],z['y'])
                    assert np.array_equal(fitted.X_thresholds_,model['calibrator'].X_thresholds_)
                    assert np.array_equal(fitted.y_thresholds_,model['calibrator'].y_thresholds_)
                # Replay selection from this fold's validation predictions.
                parent=model['parent'];fr=prov['frozen'][parent];pr=prov['frozen']['Physical_control']
                episodes=[]
                for c in va:
                    bp,res=prediction_cache[parent][c.id]
                    pp,_=prediction_cache['Physical_control'][c.id]
                    episodes.append({'cell_id':c.id,'dataset':c.dataset,'domain':c.domain,'y':c.y[K:],
                                     'base':bp,'residual':res,'physical':pp})
                replay=sa.choose_parameters(episodes,model['calibrator'],vids,tids)
                assert [replay[k] for k in ['alpha','beta','weight_a','weight_parent','weight_physical']]==[selection[k] for k in ['alpha','beta','weight_a','weight_parent','weight_physical']]
                for trial_type in ['correction_trials','fusion_trials']:
                    for a,b in zip(replay[trial_type],selection[trial_type],strict=True):
                        assert max(abs(a[k]-b[k]) for k in a)<1e-10
                adapter_cache[seed,group]=model
                selection_counts.append({'fold':fold,'seed':seed,'group':group,
                                         **{k:selection[k] for k in ['alpha','beta','weight_a']}})
        rs=json.loads((job/'cell_results.json').read_text())
        for r in rs:
            key=(r['group'],r['seed'],r['cell_id']);assert key not in seen;seen.add(key)
            assert r['cell_id'] in tids and r['fold']==fold
            with np.load(ROOT/r['prediction_file']) as z:
                c=byid[r['cell_id']]
                assert np.array_equal(z['y'],c.y[K:]) and np.array_equal(z['cycle'],c.cycle[K:])
                m=recalculate(z['y'],z[r['prediction_key']]);points+=len(z['y'])
                if r['group'] in ['Base','Base+M1','Physical_control']:
                    rebuilt=prediction_cache[r['group']][c.id][0]
                else:
                    adapter=adapter_cache[r['seed'],r['group']]
                    b,res=prediction_cache[adapter['parent']][c.id]
                    ph,_=prediction_cache['Physical_control'][c.id]
                    rebuilt=sa.combine(b,res,ph,adapter['calibrator'],adapter['selection'])
                error=float(np.max(abs(rebuilt-z[r['prediction_key']])))
                max_prediction_replay_error=max(max_prediction_replay_error,error)
                assert error<1e-10
            for k,v in m.items():
                if v is None: assert r[k] is None
                else:
                    diff=abs(v-r[k]);max_metric_error=max(max_metric_error,diff)
                    if diff>=1e-10:
                        raise AssertionError(f'{r["fold"]} {r["cell_id"]} {r["group"]} {k}: {v} != {r[k]}')
        rows.extend(rs)
    expected_ids=set(byid)
    for group in list(sa.GROUPS)+['Physical_control']:
        seeds=sa.PROTOCOL['seeds'] if group in ['Base+M2','Base+M1+M2'] else [None]
        for seed in seeds:
            rs=[r for r in rows if r['group']==group and r['seed']==seed]
            assert {r['cell_id'] for r in rs}==expected_ids and len(rs)==len(expected_ids)
    return rows,{'status':'PASS','folds':len(folds),'unique_cells':len(byid),
                 'verified_prediction_points':points,'maximum_metric_recomputation_error':max_metric_error,
                 'maximum_prediction_replay_error':max_prediction_replay_error,
                 'min_source_keys':min(source_counts),'max_source_keys':max(source_counts),
                 'replayed_parameter_selections':len(selection_counts),
                 'max_target_query_label_change':max(x['target_query_label_change'] for x in boundary),
                 'max_prefix_change':max(x['prefix_change'] for x in boundary),
                 'summed_fold_seconds':sum(runtime)},selection_counts


def build_report(root,rows,verification,selections):
    cell_mean=[];grouped=defaultdict(list)
    for r in rows:grouped[(r['dataset'],r['domain'],r['cell_id'],r['group'])].append(r)
    for (ds,dom,cid,g),rs in grouped.items():
        cell_mean.append({'dataset':ds,'domain':dom,'cell_id':cid,'group':g,
                          **{k:float(np.mean([r[k] for r in rs if r[k] is not None]))
                             if any(r[k] is not None for r in rs) else None for k in KEYS}})
    summary=[];seed_summary=[]
    for ds in DATASETS:
        for g in sa.GROUPS+['Physical_control']:
            rs=[r for r in rows if r['dataset']==ds and r['group']==g]
            seedvals=[]
            for seed in (sa.PROTOCOL['seeds'] if g in ['Base+M2','Base+M1+M2'] else [None]):
                sr=[r for r in rs if r['seed']==seed]
                entry={'dataset':ds,'group':g,'seed':seed,'cells':len(sr),**{k:equal_domain(sr,k) for k in KEYS}}
                seed_summary.append(entry);seedvals.append(entry)
            summary.append({'dataset':ds,'group':g,'cells':len({r['cell_id'] for r in rs}),
                            'repeats':len(seedvals),**{k:float(np.mean([r[k] for r in seedvals])) for k in KEYS},
                            **{k+'_std':float(np.std([r[k] for r in seedvals],ddof=1)) if len(seedvals)>1 else 0. for k in KEYS}})
    pairings=[('Base+M1','Base'),('Base+M2','Base'),('Base+M1+M2','Base+M1'),
              ('Base+M1+M2','Base+M2'),('Base+M1+M2','Base')]
    bykey={(r['group'],r['cell_id']):r for r in cell_mean}
    paired=[];interaction=[]
    for ds in DATASETS:
        reference=[r for r in cell_mean if r['dataset']==ds and r['group']=='Base']
        for a,b in pairings:
            for k in KEYS:
                diff=[];domains=[]
                for r in reference:
                    ar,br=bykey[a,r['cell_id']],bykey[b,r['cell_id']]
                    if ar[k] is not None and br[k] is not None:
                        diff.append(ar[k]-br[k]);domains.append(r['domain'])
                paired.append({'dataset':ds,'new':a,'reference':b,'metric':k,**paired_stats(diff,domains)})
        for k in KEYS[:3]:
            diff=[];domains=[]
            for r in reference:
                cid=r['cell_id']
                diff.append(bykey['Base+M1+M2',cid][k]-bykey['Base+M1',cid][k]
                            -bykey['Base+M2',cid][k]+bykey['Base',cid][k])
                domains.append(r['domain'])
            interaction.append({'dataset':ds,'metric':k,**paired_stats(diff,domains)})
    lookup={(r['dataset'],r['group']):r for r in summary}
    all_mean=all(lookup[d,'Base+M1+M2'][k] < lookup[d,'Base+M1'][k] for d in DATASETS for k in KEYS[:3])
    all_seed=all(r[k] < lookup[r['dataset'],'Base+M1'][k]
                 for r in seed_summary if r['group']=='Base+M1+M2' for k in KEYS[:3])
    independent=all(lookup[d,'Base+M2'][k] < lookup[d,'Base'][k] for d in DATASETS for k in KEYS[:3])
    complement=all(lookup[d,'Base+M1+M2'][k] < lookup[d,'Base+M2'][k] for d in DATASETS for k in KEYS[:3])
    supported=all_mean and all_seed and independent and complement
    decision={'m2_all_dataset_mean_gate':all_mean,'m2_all_seed_gate':all_seed,
              'm2_without_m1_all_dataset_gate':independent,'full_beats_m2_only_gate':complement,
              'm2_status':'NUMERICAL_CANDIDATE_REQUIRES_CONFIRMATION' if supported else 'NOT_ACCEPTED',
              'm3_decision':'DEFER: establish M2 efficacy and complementarity first' if not supported else
                            'NOT_REQUIRED_BY_MODULE_COUNT: prioritize frozen independent confirmation; only add M3 for a demonstrated unmet need',
              'independent_final_test_completed':False,'novelty_proven':False}
    result={'verification':verification,'summary':summary,'per_seed_summary':seed_summary,
            'paired_comparisons':paired,'factor_interaction':interaction,'decision':decision,
            'selection_records':selections,'cell_results':rows,'protocol':sa.PROTOCOL}
    write_json(root/'summary.json',result)
    for name,data in [('four_group_table.csv',[r for r in summary if r['group'] in sa.GROUPS]),
                      ('paired_comparisons.csv',paired),('per_seed_summary.csv',seed_summary),('cell_metrics.csv',rows)]:
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# 第二模块纠正与四组消融','',
           '此表为修正标签泄漏、源标签预算和物理预测偏移后的开发集复核结果。前一版“第二模块正式通过”结论已撤回。',
           '', '单位：SOH 百分点（原始误差 ×100），越低越好。M2 为5个源域交叉拟合划分种子的均值±样本标准差；Base/M1 是冻结的确定性预测。',
           '先计算每个电池指标，再在域内平均、数据集内各域等权。P95列是电池P95的宏平均，不是所有预测点混合后的95%分位数。','',
           '| 数据集 | 组合 | 电池数 | MAE↓ | RMSE↓ | P95-AE↓ |',
           '|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['group'] not in sa.GROUPS:continue
        vals=[f'{100*r[k]:.4f} ± {100*r[k+"_std"]:.4f}' if r['repeats']>1 else f'{100*r[k]:.4f}' for k in KEYS[:3]]
        lines.append(f'| {r["dataset"]} | {r["group"]} | {r["cells"]} | '+ ' | '.join(vals)+' |')
    lines+=['','## 第二模块的增量证据','',
            '下表为 Full−M1 的误差差值；负数表示加入M2后更好。置信区间按电芯成对、域内分层重采样；另保存整域重采样区间，以反映域数有限的影响。','',
            '| 数据集 | 指标 | 差值（百分点） | 电芯分层95%区间 | 改善电池 |',
            '|---|---|---:|---|---:|']
    for r in paired:
        if r['new']=='Base+M1+M2' and r['reference']=='Base+M1' and r['metric'] in KEYS[:3]:
            ci=r['ci95_cell_stratified_pp']
            lines.append(f'| {r["dataset"]} | {r["metric"]} | {r["delta_pp"]:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {r["improved_cells"]}/{r["cells"]} |')
    lines+=['','## 是否需要第三模块','',
            '当前第二模块未满足验收条件，暂不开发第三模块。应先修复或替换M2，随后用相同四组消融复核。' if not supported else
            '不因模块数量而增加M3。当前候选先进行冻结后的独立确认，若仍有明确误差模式，再针对该问题测试第三模块。',
            '', '## 验证范围','',
            f'- {verification["folds"]} 折、{verification["unique_cells"]} 个不同测试电芯；每个M2组合5种划分。',
            f'- 逐点复算覆盖 {verification["verified_prediction_points"]:,} 个预测；最大指标重算误差 {verification["maximum_metric_recomputation_error"]:.3g}。',
            f'- 每折源标签 {verification["min_source_keys"]}–{verification["max_source_keys"]} 个；源域子折和校准器共享同一许可集合。',
            f'- {verification["replayed_parameter_selections"]} 次选择均用本折验证电芯重放，未跨外折汇总。',
            '- 保存源标签键、数据/模型哈希、每折参数、校准器、逐点预测和标签扰动/截短测试。',
            '- 原有M1是包括剪枝、PLS旁路和支持模式选择的组合；此消融不证明单独PLS操作的新颖性。',
            '- 三个数据集已有架构探索使用历史，这些结果仍属开发验证；没有启用HUST等保留数据集。',
            '- 统计区间和数值门槛不是论文创新性证明。',
            '', '复现命令：', '', '```bash',
            'python 模块研发/m2/strict_ablation.py --workers 3',
            'python 模块研发/m2/summarize_strict.py', '```']
    (root/'四组消融与第三模块决策.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verification':verification,'summary':summary,'decision':decision},ensure_ascii=False,indent=2))


def main():
    from threadpoolctl import threadpool_limits
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(ROOT/'模块研发/results/m2_strict_ablation_v1'))
    root=Path(ap.parse_args().root)
    with threadpool_limits(limits=1):
        rows,verification,selections=audit(root)
        build_report(root,rows,verification,selections)


if __name__=='__main__':main()
