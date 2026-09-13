"""Two prospectively declared shared-selection policies; all ablations retain parent."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,GROUPS,aggregate,error_metrics,dump_csv
from summarize_strict import paired_stats

P=json.loads((HERE/'shared_protocol.json').read_text())


def select(trials,policy):
    lookup={(r['group'],r['key'],r['weight']):r['validation_mae'] for r in trials}
    configs=sorted({(r['key'],r['weight']) for r in trials if r['weight'] in P['weights']})
    base={g:next(r['validation_mae'] for r in trials if r['group']==g and r['weight']==0) for g in GROUPS}
    scores=[]
    for key,w in configs:
        value=float(np.mean([lookup[g,key,w]/base[g] for g in GROUPS])) if policy=='balanced' else lookup['B12',key,w]
        scores.append((value,w,key))
    value,w,key=min(scores)
    return dict(key=key,weight=w,validation_objective=value)


def main():
    global P
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--stage',action='store_true');args=ap.parse_args()
    if args.stage:P=json.loads((HERE/'stage_protocol.json').read_text())
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=HERE.parent/'results'/P['source_run'];manifest=json.loads((FROZEN/'candidate.json').read_text())
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    sa.write_json(out/'run_protocol.json',dict(protocol=P,code_sha256=sa.digest(Path(__file__)),protocol_sha256=sa.digest(HERE/('stage_protocol.json' if args.stage else 'shared_protocol.json'))))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)]
    rows=[];boundary=0.;selections=[];hashes={};replay=0.
    with threadpool_limits(limits=1):
        for dest in sorted((root/'folds').iterdir()):
            fold=dest.name.replace('__',':',1);train,val,test=sa.split(cells,fold)
            old=json.loads((dest/'selection.json').read_text());assert set(old['validation_cells'])=={c.id for c in val}
            entries={r['group']:r for r in manifest['manifest'] if r['fold']==fold}
            if args.stage:
                from stage_insertion import validation_trials,coefficient,combine
                trials,heads=validation_trials(dest,val,test,entries,P['weights'])
                old=dict(old,trials=trials)
                sa.write_json(out/'folds'/dest.name/'stage_validation_trials.json',trials)
            choices={policy:select(old['trials'],policy) for policy in P['policies']}
            sa.write_json(out/'folds'/dest.name/'selection.json',dict(choices=choices,validation_cells=old['validation_cells'],source_keys=old['source_keys']))
            selections.append(dict(fold=fold,choices=choices))
            models={key:joblib.load(dest/(key+'.joblib')) for key in {s['key'] for s in choices.values()}}
            assert all(set(m.source_keys)=={tuple(k) for k in old['source_keys']} for m in models.values())
            for key in models:hashes[str(dest/(key+'.joblib'))]=sa.digest(dest/(key+'.joblib'))
            for c in test:
                with np.load(dest/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:]);pred={g:z[g] for g in GROUPS}
                    branch={key:model.predict(sa.inference_view(c)) for key,model in models.items()}
                    for key,model in models.items():
                        y=c.y.copy();y[sa.K:]=123;n=min(len(c.x),sa.K+3)
                        boundary=max(boundary,float(np.max(abs(branch[key]-model.predict(replace(c,y=y))))),
                            float(np.max(abs(branch[key][:n-sa.K]-model.predict(sa.prefix(c,n))))))
                    assert boundary<1e-8
                    for policy,s in choices.items():
                        w=s['weight']
                        if args.stage:
                            coeff={g:coefficient(c,heads[g],entries[GROUPS[g]]['selection']) for g in ['B2','B12']}
                            pp=combine({g:z[g] for g in GROUPS},branch[s['key']],w,coeff)
                            pred.update({g+'__'+policy:p for g,p in pp.items()})
                        else:
                            for g in GROUPS:pred[g+'3__'+policy]=(1-w)*z[g]+w*branch[s['key']]
                    path=out/'folds'/dest.name/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,y=z['y'],**pred)
                    with np.load(path) as zz:
                        for g,p in pred.items():
                            replay=max(replay,float(np.max(abs(p-zz[g]))))
                            rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(zz['y'],zz[g])))
            print(fold,'shared policies complete',flush=True)
    assert len(rows)==4380
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={};pairs=[]
    for policy in P['policies']:
        edges=[('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]
        gates[policy]={}
        for a,b in edges:
            ga=a+'__'+policy;gb=b if b in GROUPS else b+'__'+policy
            gates[policy][a+'<'+b]=all(look[ds,ga][m]<look[ds,gb][m] for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae'])
            for ds in ['XJTU','MATR','Tongji']:
                aa={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==ga};bb={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==gb}
                for metric in ['mae','rmse','p95_ae']:
                    pairs.append(dict(policy=policy,comparison=a+'-'+b,dataset=ds,metric=metric,
                        **paired_stats([(aa[c][metric]-bb[c][metric])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    reverse={v:k for k,v in GROUPS.items()}
    parent_error=max(abs(look[r['dataset'],reverse[r['group']]][m]-100*r[m]) for r in manifest['table'] for m in ['mae','rmse','p95_ae'])
    assert parent_error<1e-8 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    assert all(sa.digest(path)==h for path,h in hashes.items())
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,selections=selections,
        paired_comparisons=pairs,max_boundary_error=boundary,max_stored_prediction_error=replay,max_frozen_parent_metric_error=parent_error,
        source_model_hashes=hashes,cells=365,folds=21))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
