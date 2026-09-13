"""Descriptive error-position audit. Relative lifetime is NEVER a model input."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import joblib
import numpy as np
import repair_diagnostics as rd


def stages(indices, length):
    if length < 1: raise ValueError('Empty query trajectory')
    return np.minimum(np.asarray(indices, dtype=int)*3//length, 2)


def summarize(rows, keys, metrics, nested=False):
    buckets=defaultdict(list)
    for r in rows: buckets[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key, rr in sorted(buckets.items()):
        units=defaultdict(list)
        for r in rr: units[(r['fold'],r['domain'])].append(r)
        domain_means={u:{m:float(np.mean([r[m] for r in group])) for m in metrics}
                      for u,group in units.items()}
        if nested:
            folds=defaultdict(list)
            for (fold,domain), values in domain_means.items(): folds[fold].append(values)
            values=[{m:float(np.mean([v[m] for v in vv])) for m in metrics} for vv in folds.values()]
        else: values=list(domain_means.values())
        result.append(dict(zip(keys,key), rows=len(rr), **{m:float(np.mean([v[m] for v in values])) for m in metrics}))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    screens={
        'v8':rd.sa.ROOT/'模块研发/results/m2_reference_raw_backbone_screen_v1',
        'v9':rd.sa.ROOT/'模块研发/results/m2_reference_position_risk_screen_v1'}
    cells=[c for d in rd.sa.PROTOCOL['datasets'] for c in rd.sa.load_cells(d)]
    target_rows=[];tail_rows=[];source_rows=[];inputs=[]
    for version,root in screens.items():
        protocol=json.loads((root/'protocol.json').read_text())
        assert protocol['status']=='COMPLETE' and len(protocol['variants'])==1
        variant=protocol['variants'][0]
        inputs.append(dict(version=version,screen=str(root),protocol_sha256=rd.sa.digest(root/'protocol.json')))
        count=0
        for fold in protocol['folds']:
            train,_,test=rd.sa.split(cells,fold)
            folder=root/'folds'/fold.replace(':','__')/'seed_0'
            for c in test:
                pred={}
                for group in ['Base+M2','Base+M1+M2']:
                    with np.load(folder/group/(c.id+'.npz')) as z:
                        assert np.array_equal(z['y'],c.y[rd.sa.K:])
                        pred[group]=z[variant]
                truth=c.y[rd.sa.K:];n=len(truth);ss=stages(np.arange(n),n)
                for group,p in pred.items():
                    error=p-truth;absolute=np.abs(error)
                    top=np.argsort(absolute,kind='stable')[-max(1,int(np.ceil(.05*n))):]
                    for stage in range(3):
                        mask=ss==stage
                        target_rows.append(dict(version=version,fold=fold,dataset=c.dataset,domain=c.domain,
                            cell_id=c.id,group=group,stage=stage,mae_pp=float(absolute[mask].mean()*100),
                            bias_pp=float(error[mask].mean()*100),p95_pp=float(np.quantile(absolute[mask],.95)*100)))
                        tail_rows.append(dict(version=version,fold=fold,dataset=c.dataset,domain=c.domain,
                            cell_id=c.id,group=group,stage=stage,tail_fraction=float(np.mean(ss[top]==stage))))
                count+=1
            if version!='v8': continue  # Same source OOF predictions in both screens.
            allowed=rd.sa.source_indices(train);byid={c.id:c for c in train}
            assert sum(len(v) for v in allowed.values())<=1000
            episodes=joblib.load(folder/'source_lodo_episodes.joblib')
            for parent,ee in episodes.items():
                for e in ee:
                    c=byid[e['cell_id']];q=allowed[c.id][allowed[c.id]>=rd.sa.K]
                    assert np.array_equal(e['y'],c.y[q])
                    ss=stages(q-rd.sa.K,len(c.x)-rd.sa.K)
                    for stage in range(3):
                        mask=ss==stage
                        if not mask.any(): continue
                        a=e['base'][mask]-e['y'][mask];b=e['physical'][mask]-e['y'][mask]
                        source_rows.append(dict(fold=fold,dataset=c.dataset,domain=c.domain,cell_id=c.id,
                            parent=parent,stage=stage,points=int(mask.sum()),
                            parent_mae_pp=float(np.mean(abs(a))*100),physical_mae_pp=float(np.mean(abs(b))*100),
                            parent_bias_pp=float(a.mean()*100),physical_bias_pp=float(b.mean()*100),
                            physical_wins=float(np.mean(abs(b))<np.mean(abs(a)))))
        assert count==365
    target=summarize(target_rows,['version','dataset','group','stage'],['mae_pp','bias_pp','p95_pp'])
    tail=summarize(tail_rows,['version','dataset','group','stage'],['tail_fraction'])
    source=summarize(source_rows,['dataset','parent','stage'],
        ['parent_mae_pp','physical_mae_pp','parent_bias_pp','physical_bias_pp','physical_wins'],nested=True)
    report=dict(status='COMPLETE',inputs=inputs,code_sha256=rd.sa.digest(Path(__file__)),
        limits=['Descriptive development diagnosis; no training, selection, or improvement claim.',
                'Thirds use full query length ONLY for retrospective reporting, forbidden as inference inputs.',
                'Top-5% positions refer to largest absolute errors, not necessarily end-of-life cycles.',
                'Source analysis reads only original budgeted query labels; stage samples are sparse.',
                'Target means: cells then equal outer domains. Source means: cells then source domains then outer folds.'],
        target_summary=target,tail_summary=tail,source_summary=source,
        target_rows=target_rows,tail_rows=tail_rows,source_rows=source_rows)
    rd.sa.write_json(out/'diagnosis.json',report)
    lines=['# 误差位置诊断（不是选参结果）','',*['- '+s for s in report['limits']],
           '','## 固定v8：完整组合相对独立M2的分段误差差值',
           '','| 数据集 | 查询位置段 | MAE差值 | P95差值 | 完整组合最大5%误差落入该段比例 |',
           '|---|---|---:|---:|---:|']
    lookup={(r['version'],r['dataset'],r['group'],r['stage']):r for r in target}
    tails={(r['version'],r['dataset'],r['group'],r['stage']):r for r in tail}
    for ds in rd.sa.PROTOCOL['datasets']:
        for stage in range(3):
            a=lookup['v8',ds,'Base+M1+M2',stage];b=lookup['v8',ds,'Base+M2',stage]
            t=tails['v8',ds,'Base+M1+M2',stage]['tail_fraction']*100
            lines.append(f'| {ds} | {stage+1}/3 | {a["mae_pp"]-b["mae_pp"]:+.4f} | {a["p95_pp"]-b["p95_pp"]:+.4f} | {t:.1f}% |')
    lines+=['','## 源预算内分段证据','',
            '| 数据集 | 父模型 | 查询位置段 | 父模型MAE | 物理分支MAE | 物理分支胜出比例 |',
            '|---|---|---|---:|---:|---:|']
    for r in source:
        lines.append(f'| {r["dataset"]} | {r["parent"]} | {r["stage"]+1}/3 | {r["parent_mae_pp"]:.4f} | {r["physical_mae_pp"]:.4f} | {100*r["physical_wins"]:.1f}% |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__': main()
