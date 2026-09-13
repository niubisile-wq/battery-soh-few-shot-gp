"""Package and independently replay the selected reference-risk M2 candidate."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import repair_diagnostics as rd
from reference_adapter import ReferenceRiskAdapter
from summarize_strict import paired_stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--screen',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--variant',default='source_lodo_reference_hi')
    args=ap.parse_args();root=Path(args.screen).resolve();out=Path(args.out).resolve()
    protocol=json.loads((root/'protocol.json').read_text());audit=json.loads((root/'screen_audit.json').read_text())
    assert protocol['status']=='COMPLETE' and audit['status']=='PASS'
    assert audit['source_lodo_selection_replays']==2*len(protocol['folds'])*len(protocol['variants'])
    chosen=args.variant
    assert chosen in protocol['variants']
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    shutil.copyfile(Path(__file__).with_name('state_risk.py'),snapshot/'state_risk.py')
    for p in [Path(__file__),Path(rd.__file__),Path(rd.sa.__file__),Path(__file__).with_name('reference_gate.py'),
              Path(__file__).with_name('reference_adapter.py'),Path(__file__).with_name('meta_residual.py'),
              rd.sa.M1/'data.py',rd.sa.M1/'features.py',rd.sa.M1/'gp.py',rd.sa.M1/'protocol.json',rd.sa.M1/'candidates_v1.json']:
        shutil.copyfile(p,snapshot/p.name)
    rd.sa.write_json(out/'request.json',dict(status='RUNNING',screen=str(root),variant=chosen))
    cells=[c for ds in rd.sa.PROTOCOL['datasets'] for c in rd.sa.load_cells(ds)]
    result=json.loads((root/'summary.json').read_text());records=[];max_replay=0.;max_boundary=0.
    byrow={(r['fold'],r['group'],r['cell_id']):r for r in result['rows'] if r['variant']==chosen}
    manifest=[]
    with threadpool_limits(limits=1):
        for fold in protocol['folds']:
            tr,va,te=rd.sa.split(cells,fold)
            job=root/'folds'/fold.replace(':','__')/'seed_0'
            oldjob=rd.OLD/'folds'/fold.replace(':','__')
            prov=json.loads((oldjob/'provenance.json').read_text())
            assets={}
            for group in rd.sa.JOBS:
                f=prov['frozen'][group];p=rd.sa.ROOT/f['job']/'model.joblib'
                assert rd.sa.digest(p)==f['model_sha256'];assets[group]=joblib.load(p)
            for parent,group in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
                head=joblib.load(job/(group+'_reference_heads.joblib'))
                selection=json.loads((job/(group+'_selection.json')).read_text())[chosen]['chosen']
                adapter=ReferenceRiskAdapter(assets[parent],assets['Physical_control'],prov['frozen'][parent]['mode'],
                    prov['frozen']['Physical_control']['mode'],head,selection,
                    raw_model=assets['Base'] if head.get('raw_reference',False) else None,
                    raw_mode=prov['frozen']['Base']['mode'] if head.get('raw_reference',False) else None)
                target=out/'folds'/fold.replace(':','__')/group;target.mkdir(parents=True)
                modelpath=target/'adapter.joblib';joblib.dump(adapter,modelpath,compress=3)
                adapter=joblib.load(modelpath)
                for c in te:
                    expected_path=job/group/(c.id+'.npz')
                    with np.load(expected_path) as z: expected=z[chosen]
                    p=adapter.predict(c);error=float(np.max(abs(p-expected)));max_replay=max(max_replay,error)
                    assert error<1e-10
                    # All target cells: query labels and future signal availability.
                    y=c.y.copy();y[rd.sa.K:]=123.
                    changed=adapter.predict(replace(c,y=y))
                    short=adapter.predict(rd.sa.prefix(c,rd.sa.K+min(3,len(p))))
                    boundary=max(float(np.max(abs(changed-p))),float(np.max(abs(short-p[:len(short)]))))
                    max_boundary=max(max_boundary,boundary);assert boundary<1e-8
                    metrics=rd.sa.metrics(c.y[rd.sa.K:],p)
                    old=byrow[fold,group,c.id]
                    for k in rd.KEYS: assert abs(metrics[k]-old[k])<1e-12
                    records.append(dict(fold=fold,dataset=c.dataset,domain=c.domain,cell_id=c.id,group=group,**metrics))
                manifest.append(dict(fold=fold,group=group,artifact=str(modelpath.relative_to(out)),
                    sha256=rd.sa.digest(modelpath),source_keys=prov['source_keys'],
                    selection=selection,screen_head_sha256=rd.sa.digest(job/(group+'_reference_heads.joblib'))))
            print(fold,'inference replay passed',flush=True)
    # Four groups use the same outer cells; no repeated deterministic controls.
    for parent,srcgroup in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
        records += [dict(r,group=parent) for r in result['rows'] if r['group']==srcgroup and r['variant']=='parent']
    table=[];pairs=[]
    for ds in rd.sa.PROTOCOL['datasets']:
        for group in ['Base','Base+M1','Base+M2','Base+M1+M2']:
            rows=[r for r in records if r['dataset']==ds and r['group']==group]
            assert len(rows)==rd.sa.PROTOCOL['expected_cells'][ds]
            table.append(dict(dataset=ds,group=group,cells=len(rows),**{k:rd.sa.macro(rows,k) for k in rd.KEYS}))
        for new,old in [('Base+M2','Base'),('Base+M1+M2','Base+M1'),('Base+M1+M2','Base+M2')]:
            a={r['cell_id']:r for r in records if r['dataset']==ds and r['group']==new}
            b={r['cell_id']:r for r in records if r['dataset']==ds and r['group']==old}
            for k in rd.KEYS:
                pairs.append(dict(dataset=ds,new=new,reference=old,metric=k,
                    **paired_stats([a[c][k]-b[c][k] for c in a],[a[c]['domain'] for c in a])))
    gates={}
    for name,new,old in [('independent','Base+M2','Base'),('incremental','Base+M1+M2','Base+M1'),('complementary','Base+M1+M2','Base+M2')]:
        gates[name]=all(p['delta_pp']<0 for p in pairs if p['new']==new and p['reference']==old)
    assert all(gates.values())
    verification=dict(status='PASS',folds=21,cells=365,adapters=len(manifest),max_prediction_replay_error=max_replay,
                      max_query_label_or_future_prefix_error=max_boundary,numerical_gates=gates)
    rd.sa.write_json(out/'candidate.json',dict(status='NUMERICAL_CANDIDATE_REQUIRES_INDEPENDENT_CONFIRMATION',
        variant=chosen,verification=verification,table=table,paired_comparisons=pairs,manifest=manifest,
        limits=['Already explored development datasets; not independent final-test evidence.',
                'Deterministic source-domain cross-fitting; repeating identical runs is not seed robustness.',
                'Paired cell and whole-domain confidence intervals must accompany point estimates.',
                'No novelty or publication acceptance claim is established by numerical gates.']))
    lines=['# M2参考信号风险门控：冻结研发候选','','开发数据四组数值验收通过；独立最终确认与创新性论证仍待完成。',
        '','| 数据集 | 组合 | MAE | RMSE | P95 |','|---|---|---:|---:|---:|']
    for r in table: lines.append('| '+' | '.join([r['dataset'],r['group']]+[f'{100*r[k]:.4f}' for k in rd.KEYS])+' |')
    independent=next(p for p in pairs if p['dataset']=='XJTU' and p['new']=='Base+M2' and p['reference']=='Base' and p['metric']=='mae')
    ci=independent['ci95_domain_bootstrap_pp']
    lines+=['','单位SOH百分点。所有数据集三项均值均满足独立增益、叠加增益与互补性。',
            f'XJTU独立MAE差值（新−旧）为{independent["delta_pp"]:+.6f}个百分点；整域95%区间[{ci[0]:+.6f},{ci[1]:+.6f}]。须保留跨工况不确定性说明。',
            '所有365电芯的两组新模型已从封存适配器重放，并检查查询标签扰动与未来信号前缀不变性。',
            '完整成对不确定性、源码快照及每折模型文件见本目录candidate.json与folds/。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    rd.sa.write_json(out/'request.json',dict(status='COMPLETE',screen=str(root),variant=chosen,verification=verification))
    print(json.dumps(verification),flush=True)


if __name__=='__main__': main()
