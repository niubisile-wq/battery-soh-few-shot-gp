"""Source-only nested error-mapping probe, with fixed outer GP configurations."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import repair_diagnostics as rd
from features import hi

METHODS=['prior','reference','delta','dual']
FLOOR=np.r_[np.full(7,1e-6),np.full(7,1e-4),np.full(7,1e-3)]


def signal_views(cell,query):
    ref=hi(cell.x[:rd.sa.K]).mean(0)
    delta=hi(cell.x[query])-ref
    reference=np.broadcast_to(ref,delta.shape).copy()
    return dict(reference=reference,delta=delta,dual=np.c_[reference,delta])


def balanced_weights(records):
    domains=defaultdict(list)
    for r in records: domains[r['domain']].append(r)
    weights=[]
    for r in records:
        weights.append(np.full(len(r['y']),1/len(domains)/len(domains[r['domain']])/len(r['y'])))
    return np.concatenate(weights)


def learn_predict(train,validation,method,views):
    w=balanced_weights(train)
    gain=np.concatenate([abs(e['base']-e['y'])-abs(e['physical']-e['y']) for e in train])
    prior=float(w@gain)
    if method=='prior': return [np.full(len(e['y']),prior) for e in validation],dict(prior=prior)
    x=np.concatenate([views[e['cell_id']][method] for e in train])
    floor=np.tile(FLOOR,2) if method=='dual' else FLOOR
    keep=x.std(0)>floor
    if not keep.any(): return [np.full(len(e['y']),prior) for e in validation],dict(prior=prior,degenerate=True)
    scaler=StandardScaler().fit(x[:,keep],sample_weight=w)
    z=scaler.transform(x[:,keep]);gamma=1/z.shape[1]
    model=KernelRidge(alpha=.01,kernel='rbf',gamma=gamma).fit(z,gain-prior,sample_weight=w)
    prediction=[prior+model.predict(scaler.transform(views[e['cell_id']][method][:,keep])) for e in validation]
    return prediction,dict(prior=prior,keep=keep,scaler=scaler,model=model)


def run_fold(task):
    fold,out=task;out=Path(out);folder=out/'folds'/fold.replace(':','__');folder.mkdir(parents=True)
    with threadpool_limits(limits=1):
        cells=rd.sa.load_cells('MATR');train,_,_=rd.sa.split(cells,fold)
        byid={c.id:c for c in train};allowed=rd.sa.source_indices(train)
        prov=json.loads((rd.OLD/'folds'/fold.replace(':','__')/'provenance.json').read_text())
        keys={(c.id,int(j)) for c in train for j in allowed[c.id]}
        assert keys=={tuple(k) for k in prov['source_keys']} and len(keys)<=1000
        rd.INNER_CACHE=str(rd.sa.ROOT/'模块研发/results/m2_source_lodo_screen_v1')
        original=rd.source_lodo_episodes(train,prov,folder)
        domains=sorted({c.domain for c in train});assert len(domains)==3
        views={c.id:signal_views(c,allowed[c.id][allowed[c.id]>=rd.sa.K]) for c in train}
        pairs={};fits=[]
        for excluded in combinations(domains,2):
            fit_cells=[c for c in train if c.domain not in excluded]
            assert fit_cells and all(c.domain not in excluded for c in fit_cells)
            indices={c.id:allowed[c.id] for c in fit_cells}
            visible=[rd.sa.source_view(c,indices[c.id]) for c in fit_cells]
            models={};where=folder/('exclude_'+'__'.join(excluded));where.mkdir()
            for group in rd.sa.JOBS:
                f=prov['frozen'][group]
                model=rd.sa.GPModel(rd.sa.SPECS[f['candidate']],f['config']).fit(visible,indices)
                assert len(model.gp.X_train_)==sum(len(v) for v in indices.values())
                mp=where/(group+'.joblib');joblib.dump(model,mp,compress=3);models[group]=model
                fits.append(dict(excluded=list(excluded),group=group,fit_keys=[[c.id,int(j)] for c in fit_cells for j in indices[c.id]],
                    artifact=str(mp.relative_to(out)),sha256=rd.sa.digest(mp),candidate=f['candidate'],config=f['config'],mode=f['mode']))
            episodes={g:[] for g in ['Base','Base+M1']}
            for c in train:
                if c.domain not in excluded: continue
                q=allowed[c.id][allowed[c.id]>=rd.sa.K];index=np.r_[np.arange(rd.sa.K),q]
                sparse=replace(c,x=c.x[index],y=c.y[index],cycle=c.cycle[index])
                sparse=rd.sa.inference_view(sparse)
                predictions={g:m.predict(sparse,prov['frozen'][g]['mode']) for g,m in models.items()}
                for g in episodes:
                    episodes[g].append(dict(cell_id=c.id,dataset=c.dataset,domain=c.domain,
                        query_indices=q.tolist(),y=c.y[q],base=predictions[g],physical=predictions['Physical_control']))
            joblib.dump(episodes,where/'episodes.joblib',compress=3);pairs[excluded]=episodes
            print(fold,'excluded',excluded,'base GPs refit',flush=True)
        rows=[];mappings=[]
        for held in domains:
            for parent in ['Base','Base+M1']:
                fit_records=[]
                for d in domains:
                    if d==held: continue
                    pair=tuple(sorted([held,d]))
                    fit_records.extend(e for e in pairs[pair][parent] if e['domain']==d)
                validation=[e for e in original[parent] if e['domain']==held]
                assert not {e['cell_id'] for e in fit_records}&{e['cell_id'] for e in validation}
                assert all(e['domain']!=held for e in fit_records)
                for e in fit_records+validation:
                    c=byid[e['cell_id']];q=allowed[c.id][allowed[c.id]>=rd.sa.K]
                    assert np.array_equal(e['y'],c.y[q])
                for method in METHODS:
                    predicted,model=learn_predict(fit_records,validation,method,views)
                    artifact=folder/f'{held}__{parent}__{method}.joblib';joblib.dump(model,artifact,compress=3)
                    mappings.append(dict(held=held,parent=parent,method=method,
                        train_ids=[e['cell_id'] for e in fit_records],validation_ids=[e['cell_id'] for e in validation],
                        artifact=str(artifact.relative_to(out)),sha256=rd.sa.digest(artifact)))
                    for e,p in zip(validation,predicted,strict=True):
                        a=abs(e['base']-e['y']);b=abs(e['physical']-e['y']);gain=a-b
                        chosen=np.where(p>0,e['physical'],e['base']);error=abs(chosen-e['y'])
                        rows.append(dict(fold=fold,domain=held,parent=parent,method=method,cell_id=e['cell_id'],
                            n=len(p),gain_mae_pp=float(abs(p-gain).mean()*100),selection_mae_pp=float(error.mean()*100),
                            oracle_mae_pp=float(np.minimum(a,b).mean()*100),regret_pp=float((error-np.minimum(a,b)).mean()*100),
                            sign_accuracy=float(np.mean((p>0)==(gain>0)))))
        result=dict(status='COMPLETE',fold=fold,source_keys=sorted(keys),fits=fits,mappings=mappings,rows=rows)
        rd.sa.write_json(folder/'result.json',result)
        return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in [Path(__file__),Path(rd.__file__),Path(rd.sa.__file__),rd.sa.M1/'gp.py',rd.sa.M1/'features.py',rd.sa.M1/'data.py']:
        shutil.copyfile(p,snapshot/p.name)
    folds=sorted({c.dataset+':'+c.domain for c in rd.sa.load_cells('MATR')})
    protocol=dict(status='RUNNING',folds=folds,methods=METHODS,
        purpose='MATR source-only pilot for current-signal reliability features; no outer target scores.',
        fitting='Error-mapping train errors use base GPs excluding BOTH meta-held and predicted source domains. Validation errors reuse source LODO GPs excluding meta-held domain.',
        labels='Original outer source keys only, never replenished; target support K10 only in prediction.',
        limits=['Frozen outer architecture/configuration inherited; this does not nest configuration selection.',
                'This is a source pilot, not a new baseline comparison or independent final-data confirmation.',
                'Sparse source-query errors and a two-expert selector do not prove full M2 improvement.'],
        parameters=dict(alpha=.01,gamma='1 / number of resolved standardized features',weights='equal source domains, cells, then queries'))
    rd.sa.write_json(out/'protocol.json',protocol)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run_fold,(fold,str(out))) for fold in folds]): results.append(f.result())
    rows=[r for result in results for r in result['rows']];summary=[]
    for parent in ['Base','Base+M1']:
        for method in METHODS:
            rr=[r for r in rows if r['parent']==parent and r['method']==method];groups=defaultdict(list)
            for r in rr: groups[r['fold'],r['domain']].append(r)
            metrics={k:float(np.mean([np.mean([r[k] for r in g]) for g in groups.values()]))
                     for k in ['gain_mae_pp','selection_mae_pp','oracle_mae_pp','regret_pp','sign_accuracy']}
            summary.append(dict(parent=parent,method=method,cell_episodes=len(rr),**metrics))
    rd.sa.write_json(out/'summary.json',dict(status='COMPLETE',summary=summary,rows=rows))
    protocol['status']='COMPLETE';rd.sa.write_json(out/'protocol.json',protocol)
    lines=['# 源预算内嵌套信号诊断（MATR）','','不是外层测试结果，也不是完整M2的改善证据。基础模型配置继承原冻结配置。',
           '','| 父模型 | 输入 | 增益估计MAE | 选择后MAE | 相对逐点理想选择的损失 | 符号准确率 |',
           '|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["parent"]} | {r["method"]} | {r["gain_mae_pp"]:.4f} | {r["selection_mae_pp"]:.4f} | {r["regret_pp"]:.4f} | {100*r["sign_accuracy"]:.1f}% |')
    lines+=['','误差单位SOH百分点。reference为支持段HI均值，delta为当前HI减支持段均值，dual为二者拼接。',
            '逐点理想选择仅是源标签上的诊断下界，不可作为可部署模型或最终性能。']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__': main()
