"""Second M3 screen: source-LODO residual transfer, with eight frozen-parent groups."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,SCREEN,GROUPS,aggregate,error_metrics,dump_csv
from residual import ResidualModel

P=json.loads((HERE/'residual_protocol.json').read_text())


def fold_run(fold,out):
    with threadpool_limits(limits=1):
        cells=sa.load_cells(fold.split(':')[0]);tr,va,te=sa.split(cells,fold)
        name=fold.replace(':','__');dest=Path(out)/'folds'/name;dest.mkdir(parents=True,exist_ok=False)
        job=SCREEN/name/'seed_0';manifest=json.loads((FROZEN/'candidate.json').read_text())
        entries={r['group']:r for r in manifest['manifest'] if r['fold']==fold}
        allowed=defaultdict(list)
        for cid,j in entries['Base+M2']['source_keys']:allowed[cid].append(j)
        budget={(cid,j) for cid,idx in allowed.items() for j in idx};byid={c.id:c for c in tr}
        assert set(allowed)==set(byid)
        audits=json.loads((job/'source_lodo_audit.json').read_text())
        for a in audits:
            fit={tuple(k) for k in a['fit_keys']};held={tuple(k) for k in a['held_keys']}
            query={tuple(k) for k in a['query_keys']}
            assert fit|held==budget and not fit&held and query=={k for k in held if k[1]>=sa.K}
            assert all(byid[cid].domain!=a['domain'] for cid,j in fit)
            assert all(byid[cid].domain==a['domain'] for cid,j in held)
        source=joblib.load(job/'source_lodo_episodes.joblib')
        cached={}
        for group in ['Base+M2','Base+M1+M2']:
            ee=joblib.load(job/(group+'_selection_episodes.joblib'))
            sa.check_validation(ee,[c.id for c in va],[c.id for c in te])
            cached[group]={e['cell_id']:e for e in ee}
        vp={}
        for c in va:
            vp[c.id]={'B':cached['Base+M2'][c.id]['base'],'B1':cached['Base+M1+M2'][c.id]['base']}
            for short,group in [('B2','Base+M2'),('B12','Base+M1+M2')]:
                s=entries[group]['selection'];vp[c.id][short]=cached[group][c.id]['reference_predictions'][s['reference_view']][s['reference_index']]
        models={};choices={};trials=[]
        for family in ['B','B1']:
            assert {e['cell_id'] for e in source[GROUPS[family]]}==set(byid)
            for view in P['views']:
                for learner in P['learners']:
                    key=family+'__'+view+'__'+learner
                    models[key]=ResidualModel(view,learner).fit(source[GROUPS[family]],
                        [sa.source_view(c,allowed[c.id]) for c in tr],allowed)
                    joblib.dump(models[key],dest/(key+'.joblib'))
        for group in GROUPS:
            family='B1' if '1' in group else 'B';scores=[]
            for key,model in models.items():
                if not key.startswith(family+'__'):continue
                residual={c.id:np.clip(model.predict(sa.inference_view(c),vp[c.id][group]),-P['clip'],P['clip']) for c in va}
                for w in P['weights']:
                    rr=[dict(dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(
                        vp[c.id][group]+w*residual[c.id]-c.y[sa.K:])))) for c in va]
                    loss=sa.macro(rr);scores.append((loss,w,key));trials.append(dict(group=group,key=key,weight=w,validation_mae=loss))
            loss,w,key=min(scores);choices[group]=dict(key=key,weight=w,validation_mae=loss)
        sa.write_json(dest/'selection.json',dict(choices=choices,trials=trials,source_keys=sorted(budget),validation_cells=[c.id for c in va]))
        rows=[];boundary=0.
        for c in te:
            pred={}
            for short,group in [('B2','Base+M2'),('B12','Base+M1+M2')]:
                with np.load(job/group/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:])
                    pred[short]=z[manifest['variant']].copy();pred['B' if short=='B2' else 'B1']=z['parent'].copy()
            for group,s in choices.items():
                model=models[s['key']];p=pred[group];r=model.predict(sa.inference_view(c),p)
                changed=c.y.copy();changed[sa.K:]=123;n=min(len(c.x),sa.K+3)
                boundary=max(boundary,float(np.max(abs(r-model.predict(replace(c,y=changed),p)))),
                    float(np.max(abs(r[:n-sa.K]-model.predict(sa.prefix(c,n),p[:n-sa.K])))))
                assert boundary<1e-8
                pred[group+'3']=p+s['weight']*np.clip(r,-P['clip'],P['clip'])
            path=dest/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,y=c.y[sa.K:],**pred)
            for group,p in pred.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=group,**error_metrics(c.y[sa.K:],p)))
        sa.write_json(dest/'result.json',dict(fold=fold,rows=rows,max_boundary_error=boundary,choices=choices))
        print(fold,'complete',flush=True);return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((FROZEN/'candidate.json').read_text())
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    paths=[HERE/'residual.py',HERE/'geometry.py',HERE/'residual_protocol.json',Path(__file__)]
    sa.write_json(out/'run_protocol.json',dict(protocol=P,hashes={p.name:sa.digest(p) for p in paths}))
    folds=sorted({r['fold'] for r in manifest['manifest']});rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in as_completed([pool.submit(fold_run,f,str(out)) for f in folds]):rows.extend(task.result())
    assert len(rows)==2920
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    gates={a+'<'+b:all(look[ds,a][m]<look[ds,b][m] for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae'])
        for a,b in [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]}
    reverse={v:k for k,v in GROUPS.items()}
    parent_error=max(abs(look[r['dataset'],reverse[r['group']]][m]-r[m]*100) for r in manifest['table'] for m in ['mae','rmse','p95_ae'])
    assert parent_error<1e-8 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,max_frozen_parent_metric_error=parent_error,cells=365,folds=21))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
