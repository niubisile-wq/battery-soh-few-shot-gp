"""Retrospective source-budget expert capacity; never a deployable oracle."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import reference_gate as rg
import repair_diagnostics as rd


def capacity(episode,params):
    prediction=np.stack([rg.expert_prediction(episode,p) for p in params])
    y=episode['y'];error=abs(prediction-y)
    best=int(error.mean(1).argmin())
    # Per-query convex hull is a larger, label-informed class than the grid.
    hull=np.clip(y,prediction.min(0),prediction.max(0))
    result=dict(parent_mae_pp=float(abs(episode['base']-y).mean()*100),
                cell_oracle_mae_pp=float(error[best].mean()*100),
                point_oracle_mae_pp=float(error.min(0).mean()*100),
                hull_oracle_mae_pp=float(abs(hull-y).mean()*100),
                outside_hull_fraction=float(np.mean((y<prediction.min(0))|(y>prediction.max(0)))),
                best_parameter=params[best])
    assert result['hull_oracle_mae_pp']<=result['point_oracle_mae_pp']+1e-9
    assert result['point_oracle_mae_pp']<=result['cell_oracle_mae_pp']+1e-9
    assert result['cell_oracle_mae_pp']<=result['parent_mae_pp']+1e-9
    return result


def aggregate(rows):
    summary=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        for parent in ['Base','Base+M1']:
            grouped=defaultdict(list)
            for r in rows:
                if r['dataset']==dataset and r['parent']==parent: grouped[r['fold'],r['domain']].append(r)
            values={k:float(np.mean([np.mean([r[k] for r in g]) for g in grouped.values()])) for k in
                    ['parent_mae_pp','cell_oracle_mae_pp','point_oracle_mae_pp','hull_oracle_mae_pp','outside_hull_fraction']}
            summary.append(dict(dataset=dataset,parent=parent,episodes=sum(map(len,grouped.values())),**values))
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sa=rd.sa;candidate=sa.ROOT/'模块研发/results/m2_reference_candidate_v8'
    request=json.loads((candidate/'request.json').read_text());screen=Path(request['screen'])
    protocol=json.loads((screen/'protocol.json').read_text());rows=[];inputs=[]
    for dataset in sa.PROTOCOL['datasets']:
        cells=sa.load_cells(dataset)
        for fold in protocol['folds']:
            if not fold.startswith(dataset+':'): continue
            train,_,_=sa.split(cells,fold);byid={c.id:c for c in train};allowed=sa.source_indices(train)
            job=screen/'folds'/fold.replace(':','__')/'seed_0'
            source=job/'source_lodo_episodes.joblib';original=joblib.load(source)
            inputs.append(dict(path=str(source),sha256=sa.digest(source)))
            for parent,group in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
                path=job/(group+'_reference_heads.joblib');head=joblib.load(path)
                inputs.append(dict(path=str(path),sha256=sa.digest(path)))
                episodes=rg.with_raw_reference(original[parent],original['Base'])
                assert [e['cell_id'] for e in episodes]==head['source_ids']
                for i,e in enumerate(episodes):
                    c=byid[e['cell_id']];q=allowed[c.id][allowed[c.id]>=sa.K]
                    np.testing.assert_array_equal(e['y'],c.y[q])
                    risk=np.array([rg.expert_risk(e,p) for p in head['params']])
                    np.testing.assert_allclose(risk,head['costs'][i],rtol=0,atol=1e-12)
                    rows.append(dict(dataset=dataset,fold=fold,domain=c.domain,parent=parent,cell_id=c.id,
                                     query_indices=q.tolist(),**capacity(e,head['params'])))
    summary=aggregate(rows)
    sa.write_json(out/'diagnosis.json',dict(status='COMPLETE',rows=rows,summary=summary,inputs=inputs,
        limits=['Only original source-budget OOF query errors; source predictions inherit frozen outer configurations.',
                'All oracles use scored labels to choose predictions; unavailable at deployment and not validation results.',
                'Full expert bank includes raw-only bank: oracle dominance is structural, not proof of module complementarity.',
                'This measures MAE capacity, not deployable selection loss or proof of P95 repair.']))
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    lines=['# 源预算专家能力诊断','','全部理想选择使用了答案，仅作能力下界，不是可部署性能或论文测试表。',
           '','| 数据集 | 父模型 | 父模型MAE | 电芯固定专家理想MAE | 逐点专家理想MAE | 逐点凸组合理想MAE | 答案在候选范围外 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["dataset"]} | {r["parent"]} | {r["parent_mae_pp"]:.4f} | {r["cell_oracle_mae_pp"]:.4f} | {r["point_oracle_mae_pp"]:.4f} | {r["hull_oracle_mae_pp"]:.4f} | {r["outside_hull_fraction"]:.1%} |')
    lines+=['','MAE单位SOH百分点；先电芯、再源域/外层折等权，重复电芯不视作独立样本。',
            '完整专家库包含原始骨干专家，理想选择更好不能证明真实互补；本表不衡量实际选路损失。']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__': main()
