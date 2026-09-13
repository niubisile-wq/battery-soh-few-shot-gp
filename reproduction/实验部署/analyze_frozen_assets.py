"""Post hoc, no-fit audit of frozen evidence. Writes only to this review directory."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
CAND = ROOT / '模块研发/results/m4_random_function_candidate_v1'
STATS = ROOT / '补充实验_20260910/statistics'
EXTERNAL = ROOT / '补充实验_20260910/external/confirmation'
DATASETS = ['XJTU', 'MATR', 'Tongji']
STAGES = ['B', 'B1', 'B12', 'B123', 'B1234']
METRICS = ['mae', 'rmse', 'p95_ae']


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def group_name(bits):
    return 'B' + ''.join(str(i + 1) for i in range(4) if bits & (1 << i))


def summarize_errors(pred, y):
    err = (np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)) * 100
    return dict(mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))),
                p95_ae=float(np.quantile(np.abs(err), .95)))


def inventory():
    counts, sizes, modes, scopes = Counter(), Counter(), Counter(), Counter()
    hashes, scope_hashes = defaultdict(set), defaultdict(set)
    by_path, errors, series = {}, [], defaultdict(lambda: {'files': 0, 'bytes': 0, 'suffixes': Counter(), 'records': []})
    with (OUT / '逐文件程序审查.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            by_path[r['path']] = r.get('sha256')
            ext, scope = r['suffix'], r['scope']
            counts[ext] += 1; sizes[ext] += r['size_bytes']
            modes[r['review_mode']] += 1; scopes[scope] += 1
            hashes[ext].add(r.get('sha256')); scope_hashes[scope].add(r.get('sha256'))
            if not r.get('read_ok'): errors.append(r)
            parts = Path(r['path']).parts
            if 'results' in parts:
                i = parts.index('results')
                if i + 1 < len(parts):
                    key = '/'.join(parts[:i+2]); s = series[key]
                    s['files'] += 1; s['bytes'] += r['size_bytes']; s['suffixes'][ext] += 1
                    if len(parts) == i + 3 and ext in ['.md', '.json', '.csv']:
                        s['records'].append({k: v for k, v in r.items() if k not in ['sha256', 'scope', 'size_bytes', 'review_mode', 'suffix']})
    save_json('实验系列索引.json', series)
    save_json('最终覆盖统计.json', {
        'files': sum(counts.values()), 'bytes': sum(sizes.values()), 'read_errors': errors,
        'suffix_count': counts, 'unique_hashes_by_suffix': {k: len(v) for k, v in hashes.items()},
        'review_modes': modes, 'scopes': scopes, 'experiment_series': len(series),
        'limits': 'Full-file hashing and structural parsing are not semantic inspection of every tensor or every archive member.'
    })
    return by_path


def check_manifest(by_path, candidate):
    checks = []
    for kind in ['data', 'predictions', 'manifest']:
        for r in candidate[kind]:
            p = (ROOT / r['path']) if kind == 'data' else CAND / r.get('path', r.get('artifact'))
            rel = str(p.relative_to(ROOT))
            checks.append(dict(kind=kind, path=rel, expected_sha256=r['sha256'],
                               scanned_sha256=by_path.get(rel), match=by_path.get(rel) == r['sha256']))
    pd.DataFrame(checks).to_csv(OUT / '最终资产清单哈希核对.csv', index=False)
    return {'records': len(checks), 'by_kind': dict(Counter(x['kind'] for x in checks)),
            'mismatches': [x for x in checks if not x['match']]}


def recompute(candidate):
    declared = pd.read_csv(STATS / 'cell_metrics.csv')
    declared_index = declared.set_index(['dataset', 'cell_id', 'group'])
    cell_rows = []
    for meta in candidate['predictions']:
        p = CAND / meta['path']
        dataset, cell = meta['cell_id'].split('/', 1)
        domain = declared_index.loc[(dataset, cell, 'B'), 'domain']
        with np.load(p, allow_pickle=False) as z:
            y = np.asarray(z['y'], dtype=np.float64)
            low = y < .90
            for group in (group_name(i) for i in range(16)):
                pred = np.asarray(z[group], dtype=np.float64)
                assert pred.shape == y.shape and np.isfinite(pred).all() and np.isfinite(y).all(), str(p)
                row = dict(dataset=dataset, domain=domain, cell_id=cell, group=group, n=len(y), low_n=int(low.sum()))
                row.update(summarize_errors(pred, y))
                row.update({'low_' + k: v for k, v in summarize_errors(pred[low], y[low]).items()} if low.any()
                           else {'low_' + k: np.nan for k in METRICS})
                cell_rows.append(row)
    cells = pd.DataFrame(cell_rows)
    cols = METRICS + ['low_' + m for m in METRICS]
    join = cells.merge(declared, on=['dataset', 'domain', 'cell_id', 'group'], suffixes=('_recalc', '_declared'), validate='one_to_one')
    assert len(join) == len(declared) == 5840
    delta = {k: float(np.nanmax(np.abs(join[k+'_recalc'] - join[k+'_declared']))) for k in cols + ['n', 'low_n']}
    domain = cells.groupby(['dataset', 'domain', 'group'], sort=False)[cols].mean().reset_index()
    aggregate = domain.groupby(['dataset', 'group'], sort=False)[cols].mean().reset_index()
    expected = pd.read_csv(STATS / 'full_ablation_16_groups.csv')
    agg_join = aggregate.merge(expected, on=['dataset', 'group'], suffixes=('_recalc', '_declared'), validate='one_to_one')
    agg_delta = {k: float(np.nanmax(np.abs(agg_join[k+'_recalc'] - agg_join[k+'_declared']))) for k in cols}
    cells.to_csv(OUT / '最终5840条电芯指标独立复算.csv', index=False)
    domain.to_csv(OUT / '21工况完整消融复算.csv', index=False)
    aggregate.to_csv(OUT / '48组总体指标独立复算.csv', index=False)
    assert max(delta.values()) < 1e-9 and max(agg_delta.values()) < 1e-9, (delta, agg_delta)
    return cells, domain, aggregate, {'cell_rows': len(cells), 'query_points': int(cells[cells.group == 'B'].n.sum()),
                                     'cell_max_abs_error_pp': delta, 'aggregate_max_abs_error_pp': agg_delta,
                                     'limits': 'Recalculation from saved predictions, not inference replay or model refitting.'}


def ablations(aggregate):
    idx = aggregate.set_index(['dataset', 'group'])
    edges, module_rows, shapley_rows = [], [], []
    for ds in DATASETS:
        for m in range(4):
            marginal = []
            for bits in range(16):
                if bits & (1 << m): continue
                parent, child = group_name(bits), group_name(bits | (1 << m))
                row = dict(dataset=ds, module='M'+str(m+1), parent=parent, child=child)
                row.update({metric+'_gain_pp': float(idx.loc[(ds, parent), metric] - idx.loc[(ds, child), metric]) for metric in METRICS})
                edges.append(row); marginal.append((bits.bit_count(), row))
            one_out = group_name(15 ^ (1 << m))
            for metric in METRICS:
                module_rows.append(dict(dataset=ds, module='M'+str(m+1), metric=metric,
                    mean_over_8_parents_gain_pp=float(np.mean([row[metric+'_gain_pp'] for _, row in marginal])),
                    full_model_removal_loss_pp=float(idx.loc[(ds, one_out), metric] - idx.loc[(ds, 'B1234'), metric])))
                shapley_rows.append(dict(dataset=ds, module='M'+str(m+1), metric=metric,
                    factorial_shapley_gain_pp=sum(math.factorial(n)*math.factorial(3-n)/math.factorial(4)*row[metric+'_gain_pp'] for n, row in marginal)))
    pd.DataFrame(edges).to_csv(OUT / '32条消融边逐数据集效应.csv', index=False)
    pd.DataFrame(module_rows).to_csv(OUT / '四模块边际效应与移除损失.csv', index=False)
    pd.DataFrame(shapley_rows).to_csv(OUT / '完整消融描述性Shapley分配.csv', index=False)
    timing = pd.read_csv(ROOT / '模块研发/results/paper_completion_v1/efficiency/timing_comparison.csv')
    timing = timing[timing.requested_queries == 1].set_index(['dataset', 'group'])
    stages = []
    for ds in DATASETS:
        base = idx.loc[(ds, 'B'), 'mae']; full = idx.loc[(ds, 'B1234'), 'mae']
        for group in STAGES:
            mae = idx.loc[(ds, group), 'mae']
            stages.append(dict(dataset=ds, group=group, mae_pp=float(mae),
                gain_vs_B_pp=float(base-mae), relative_reduction_vs_B_pct=float((base-mae)/base*100),
                fraction_of_full_gain_pct=float((base-mae)/(base-full)*100),
                single_query_ms=float(timing.loc[(ds, group), 'mean_fold_median_ms'])))
    pd.DataFrame(stages).to_csv(OUT / '五阶段收益与推理成本.csv', index=False)
    return pd.DataFrame(stages), pd.DataFrame(module_rows), edges


def external():
    tab = pd.read_csv(EXTERNAL / 'summary/domain_mean_across_sources.csv')
    pairs = tab[tab.group=='B'][['domain', 'cells', 'mae_pp']].merge(tab[tab.group=='B1234'][['domain', 'mae_pp']], on='domain', suffixes=('_B', '_full'), validate='one_to_one')
    pairs['delta_full_minus_B_pp'] = pairs.mae_pp_full - pairs.mae_pp_B
    pairs['contribution_to_equal_protocol_mean_delta_pp'] = pairs.delta_full_minus_B_pp / 4
    pairs.to_csv(OUT / 'CALCE四协议组收益分解.csv', index=False)
    rows = []
    for p in sorted((EXTERNAL / 'folds').glob('*/predictions/*/*.npz')):
        with np.load(p, allow_pickle=False) as z:
            pred, y = np.asarray(z['pred'], dtype=np.float64), np.asarray(z['y'], dtype=np.float64)
            rows.append(dict(source_fold=p.parts[-4], cell=p.parent.name, group=p.stem, n=len(y),
                             pred_range_pp=float(np.ptp(pred)*100), pred_sd_pp=float(pred.std()*100),
                             y_range_pp=float(np.ptp(y)*100), y_sd_pp=float(y.std()*100),
                             constant_within_1e_8_pp=bool(np.ptp(pred)*100 < 1e-8),
                             **summarize_errors(pred, y)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'CALCE全部210条预测轨迹变化诊断.csv', index=False)
    check = df.copy()
    check['source_fold'] = check.source_fold.str.replace('__', ':', regex=False)
    declared = pd.read_csv(EXTERNAL / 'summary/cell_fold_metrics.csv')
    check = check.merge(declared, left_on=['source_fold','cell','group'],
                        right_on=['source_fold','cell_id','group'], validate='one_to_one')
    metric_error = {m: float((check[m]-check[m+'_pp']).abs().max()) for m in METRICS}
    assert len(check) == 210 and max(metric_error.values()) < 1e-9
    protocol_recalc = check.groupby(['source_fold','domain','group'])[METRICS].mean().groupby(['domain','group']).mean().reset_index()
    protocol_check = protocol_recalc.merge(tab, on=['domain','group'],validate='one_to_one')
    protocol_error = {m: float((protocol_check[m]-protocol_check[m+'_pp']).abs().max()) for m in METRICS}
    assert max(protocol_error.values()) < 1e-9
    base = df[df.group == 'B']
    full = df[df.group == 'B1234']
    full.groupby('cell').agg(pred_range_min=('pred_range_pp','min'), pred_range_max=('pred_range_pp','max'),
                            true_range=('y_range_pp','first'), pred_std_min=('pred_sd_pp','min'),
                            pred_std_max=('pred_sd_pp','max'), true_std=('y_sd_pp','first')).to_csv(OUT / 'CALCE完整模型轨迹范围与真实范围.csv')
    return pairs, {'prediction_files': len(df), 'cell_metric_max_error_pp': metric_error,
                   'protocol_metric_max_error_pp': protocol_error,
                   'base_constant_trajectories': int(base.constant_within_1e_8_pp.sum()),
                   'full_constant_trajectories': int(full.constant_within_1e_8_pp.sum()),
                   'trajectories_per_group': len(base), 'mean_delta_pp': float(pairs.delta_full_minus_B_pp.mean()),
                   'base_max_pred_range_pp': float(base.pred_range_pp.max()),
                   'full_max_pred_range_pp': float(full.pred_range_pp.max()),
                   'true_range_min_pp': float(full.y_range_pp.min()), 'true_range_max_pp': float(full.y_range_pp.max()),
                   'limits': 'Constant predictions describe output behavior; without a kernel replay they do not establish its unique cause. Four protocol groups are not a controlled rate experiment.'}


def controls():
    paired = pd.read_csv(ROOT / '模块研发/results/m3_waveform_audit_v1/control_comparison.csv')
    paired['comparison_type'] = 'same_config'
    selected = pd.read_csv(ROOT / '模块研发/results/m3_waveform_shrink_v1/comparison.csv')
    selected = selected[selected.group.isin(['B123__wave_absolute_pca8__0.5', 'B123__wave_pair_pca8__0.5'])].copy()
    selected['representation'] = selected.group.map({'B123__wave_absolute_pca8__0.5': 'absolute_source_selected', 'B123__wave_pair_pca8__0.5': 'paired'})
    selected['group'] = 'B123'; selected['comparison_type'] = 'separately_source_selected'
    pd.concat([paired, selected], ignore_index=True).to_csv(OUT / 'M3配对与绝对波形两类对照.csv', index=False)
    paired_intervals = pd.read_csv(STATS / 'paired_intervals.csv')
    paired_intervals[(paired_intervals.metric == 'mae') & paired_intervals.comparison.isin(['B1234-B', 'B123-B12', 'B1234-B123'])].to_csv(OUT / '主要收益及M3M4整工况区间.csv', index=False)
    fit = json.loads((ROOT / '模块研发/results/m4_random_function_fit_screen_v1/summary.json').read_text())
    fits = pd.DataFrame([r for r in fit['table'] if r['group'].startswith('B1234')])
    fits.to_csv(OUT / 'M4五个源GP初值结果.csv', index=False)
    return {'passed_edges_by_init': {k: sum(v.values()) for k, v in fit['gates'].items()},
            'full_model_mae_range_pp': {ds: float(np.ptp(fits.loc[fits.dataset==ds, 'mae'])) for ds in DATASETS},
            'limits': 'Fixed representation, kernel choice, gain, and target mode. Source GP optimization sensitivity, not whole-pipeline random retraining.'}


def figures(stages, domain, calce):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        s = stages[stages.dataset==ds]
        ax.plot(s.single_query_ms, s.mae_pp, 'o-', color='#236491')
        for _, row in s.iterrows():
            ax.annotate(row.group, (row.single_query_ms,row.mae_pp), xytext=(3,6), textcoords='offset points', fontsize=9)
        ax.set_title(ds); ax.set_xlabel('Single-query inference (ms)'); ax.set_ylabel('MAE (SOH pp)'); ax.grid(alpha=.18)
        ax.margins(x=.14, y=.22)
    fig.suptitle('Existing frozen stages: accuracy and measured inference cost', fontsize=13)
    for ext in ['png','pdf']: fig.savefig(OUT / ('图1_已有阶段的精度与成本.'+ext), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4,3.8), constrained_layout=True)
    labels = {'CS2_low_SOC_RPT_0.22A':'CS2 low-SOC RPT (n=2)', 'CX2_alternating_pulse_full_reference':'CX2 alternating pulse (n=1)',
              'PL_full_SOC_0.5C':'PL full-SOC 0.5C (n=2)', 'PL_full_SOC_2C':'PL full-SOC 2C (n=2)'}
    vals = calce.delta_full_minus_B_pp
    bars = ax.barh([labels[d] for d in calce.domain], vals, color=['#bd573b' if v>0 else '#287d7d' for v in vals])
    for bar, v in zip(bars, vals):
        ax.text(v+(.025 if v>=0 else -.025),bar.get_y()+bar.get_height()/2, f'{v:+.3f}', ha='left' if v>=0 else 'right', va='center')
    ax.axvline(0,color='#777',lw=.8); ax.set_xlim(-.9,2.25); ax.invert_yaxis()
    ax.set_xlabel('MAE change: B1234 minus B (SOH pp); negative favors B1234')
    ax.set_title(f'CALCE: equal-protocol mean change = {vals.mean():+.3f} pp\nEach protocol result averages the same six frozen source models')
    for ext in ['png','pdf']: fig.savefig(OUT / ('图2_CALCE总体均值背后的协议差异.'+ext), dpi=180)
    plt.close(fig)

    pivot = domain.pivot(index=['dataset','domain'], columns='group', values='mae').reindex(DATASETS, level=0)
    columns = [('B','B1','M1 added to B'), ('B1','B12','M2 added to B1'), ('B12','B123','M3 added to B12'), ('B123','B1234','M4 added to B123'), ('B','B1234','Full versus B')]
    values = np.column_stack([pivot[p]-pivot[c] for p,c,_ in columns])
    fig, ax = plt.subplots(figsize=(9.6,8.4), constrained_layout=True)
    vmax = max(abs(values.min()),abs(values.max()))
    im = ax.imshow(values, cmap='RdBu', vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_xticks(range(len(columns)), [x[2] for x in columns],rotation=22,ha='right')
    ax.set_yticks(range(len(pivot)), [ds+': '+dom.replace('Tongji','T') for ds,dom in pivot.index])
    for (i,j),v in np.ndenumerate(values):
        ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.6*vmax else '#111')
    ax.set_title('All 21 held-out conditions: stagewise MAE reduction (SOH pp)\nPositive values favor the added module; repeated development cohorts')
    fig.colorbar(im,ax=ax,shrink=.7,label='MAE reduction (pp)')
    for ext in ['png','pdf']: fig.savefig(OUT / ('图3_全部21工况的模块收益.'+ext), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
    for ax, cell in zip(axes.flat, ['CS2_5','CS2_6','CX2_3','PL11','PL12','PL13','PL14']):
        for i, folder in enumerate(sorted((EXTERNAL / 'folds').iterdir())):
            for group, color in [('B','#8a8a8a'), ('B1234','#237b78')]:
                with np.load(folder/'predictions'/cell/(group+'.npz'),allow_pickle=False) as z:
                    ax.plot(z['cycle'],z['pred']*100,color=color,lw=.9,alpha=.5,
                            label=group+' (six source models)' if i==0 else None)
                    if i==0 and group=='B':
                        ax.plot(z['cycle'],z['y']*100,color='#161616',lw=1.0,alpha=.8,label='Observed query SOH')
        ax.set_title(cell); ax.set_xlabel('Recorded measurement index'); ax.set_ylabel('SOH (%)'); ax.grid(alpha=.12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[7].axis('off'); axes.flat[7].legend(handles,labels,loc='center',frameon=False)
    axes.flat[8].axis('off')
    axes.flat[8].text(0,.65,'All seven eligible cells retained.\nEach source model is shown separately.\nNo prediction ensemble is evaluated.\nQuery traces follow the ten-label support.',va='top',fontsize=10,linespacing=1.6)
    fig.suptitle('Frozen CALCE predictions: nearly flat estimates despite observed capacity changes',fontsize=13)
    for ext in ['png','pdf']: fig.savefig(OUT / ('图4_CALCE全部七电芯与六源模型轨迹.'+ext),dpi=180)
    plt.close(fig)


def main():
    by_path = inventory()
    candidate = json.loads((CAND / 'candidate.json').read_text())
    manifest = check_manifest(by_path, candidate)
    assert not manifest['mismatches'], manifest
    cells, domain, aggregate, replay = recompute(candidate)
    stages, modules, edges = ablations(aggregate)
    calce, external_summary = external()
    fit_summary = controls()
    inactive = [r['fold'] for r in candidate['manifest'] if r['group']=='B1234' and r['effective_gain']==0]
    affected = cells[(cells.group=='B') & (cells.dataset+':'+cells.domain).isin(inactive)]
    summary = dict(manifest=manifest, metric_recalculation=replay,
        all_96_dataset_edges_improve_all_three=all(all(r[m+'_gain_pp']>0 for m in METRICS) for r in edges),
        m4_zero_gain_folds=inactive, m4_zero_gain_cells=len(affected), external=external_summary,
        gp_initialization=fit_summary,
        stage_summary=stages.to_dict('records'),
        limits='New results are descriptive calculations from frozen files. No training, tuning, cohort changes, ensemble construction, or manuscript edits.')
    save_json('独立复算与新增诊断摘要.json', summary)
    figures(stages, domain, calce)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
