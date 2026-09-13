"""Source-episodic residual adaptation layered on frozen SHB.

The residual map is trained only from cross-fitted source cells.  A target
cell contributes its first K labeled supports and the current query features;
future target labels are never used.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

M1=Path(__file__).resolve().parents[1]/'m1';sys.path.insert(0,str(M1))
from data import load_cells, split, source_indices, metrics  # noqa: E402
from gp import GPModel  # noqa: E402
from features import target_shift  # noqa: E402

K=10; MODES=['source_only','source_bias','posterior']; ALPHAS=[0.01,0.1,1.,10.,100.]

def macro(rows,key='mae'):
    g=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:g[(r['dataset'],r['domain'])].append(float(r[key]))
    ds=defaultdict(list)
    for (d,_),v in g.items():
        if v:ds[d].append(float(np.mean(v)))
    return float(np.mean([np.mean(v) for v in ds.values()]))

def prior(model,c):
    z=model.features.transform(c); p=model.gp.predict(z)
    if model.mean_model is not None:p=p+model.mean_model.predict(z)
    return p+target_shift(c,model.spec.get('anchored',False))

def predict_mode(model,c,mode):
    return model.predict(c,mode)

def episode_features(model,c,mode):
    p=prior(model,c); r=c.y[:K]-p[:K]
    t=np.arange(K,dtype=float); tc=t-t.mean()
    slope=float(np.dot(r,tc)/(np.dot(tc,tc)+1e-12))
    base=p[K:] if mode=='source_only' else predict_mode(model,c,mode)
    support_mean=float(p[:K].mean())
    q=np.c_[np.full(len(base),r.mean()),np.full(len(base),slope),
            np.full(len(base),r.std()),base,np.full(len(base),support_mean),
            base-support_mean]
    return q

def fit_meta(X,y,alpha):
    scaler=StandardScaler().fit(X)
    model=Ridge(alpha=alpha).fit(scaler.transform(X),y)
    return scaler,model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-root',default='模块研发/results/m1_v1/screen_v2');ap.add_argument('--out',default='模块研发/results/m2_meta_v1');ap.add_argument('--variant',choices=['residual','calibration','calibration_support'],default='residual');args=ap.parse_args()
    root=Path(args.base_root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'fold_models').mkdir(exist_ok=True)
    spec=json.loads((M1/'candidates_v1.json').read_text())['P07_pls_concat']
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in load_cells(ds)]
    folds=sorted({f'{c.dataset}:{c.domain}' for c in cells}); all_rows=[]; records=[]
    for fold in folds:
        train,val,test=split(cells,fold)
        allowed_indices=source_indices(train)
        job=root/'jobs'/f'P07_pls_concat__{fold.replace(":","__")}'
        frozen=joblib.load(job/'model.joblib'); chosen=json.loads((job/'tuning.json').read_text())['chosen']
        mode=chosen['mode']; config=chosen['config']
        # Two source-cell folds provide cross-fitted residual targets.
        ordered=sorted(train,key=lambda c:c.id); groups=[ordered[::2],ordered[1::2]]
        metaX=[];metaY=[]
        for held in groups:
            held_ids={c.id for c in held}
            fitcells=[c for c in train if c.id not in held_ids]
            idx={c.id:allowed_indices[c.id] for c in fitcells}
            cm=GPModel(spec,config).fit(fitcells,idx)
            for c in held:
                X=episode_features(cm,c,mode); base=predict_mode(cm,c,mode)
                selected=allowed_indices[c.id][allowed_indices[c.id]>=K]
                metaX.append(X[selected-K]);metaY.append(c.y[selected]-base[selected-K])
        X=np.concatenate(metaX);y=np.concatenate(metaY)
        fitted=[]
        if args.variant=='residual':
            for alpha in ALPHAS:fitted.append((alpha,*fit_meta(X,y,alpha)))
        elif args.variant=='calibration':
            calibration=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(X[:,3],X[:,3]+y)
            fitted=[(alpha,calibration,None) for alpha in [0.,0.25,0.5,0.75,1.0]]
        else:
            calibration=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(X[:,3],X[:,3]+y)
            fitted=[(alpha,beta,calibration) for alpha in [0.,0.25,0.5,0.75,1.0] for beta in [-0.5,-0.25,0.,0.25,0.5]]
        trials=[]
        for item in fitted:
            if args.variant=='calibration_support': alpha,beta,sc=item
            else: alpha,sc,rm=item; beta=0.
            rows=[]
            for c in val:
                Xc=episode_features(frozen,c,mode); base=predict_mode(frozen,c,mode)
                if args.variant=='residual': pred=base+rm.predict(sc.transform(Xc))
                elif args.variant=='calibration': pred=base+alpha*(sc.predict(base)-base)
                else: pred=base+alpha*(sc.predict(base)-base)+beta*Xc[:,0]
                rows.append({'dataset':c.dataset,'domain':c.domain,**metrics(c.y[K:],pred)})
            trials.append({'alpha':alpha,'beta':beta,'inner_mae':macro(rows),'inner_rmse':macro(rows,'rmse'),'inner_p95_ae':macro(rows,'p95_ae')})
        pick=min(trials,key=lambda z:(z['inner_mae'],z['inner_rmse'],z['inner_p95_ae']))
        if args.variant=='calibration_support': alpha,beta,sc=next(z for z in fitted if z[0]==pick['alpha'] and z[1]==pick['beta'])
        else: alpha,sc,rm=next(z for z in fitted if z[0]==pick['alpha']); beta=0.
        if args.variant=='calibration_support':
            joblib.dump({'calibration':sc,'alpha':alpha,'beta':beta,'mode':mode},out/'fold_models'/f'{fold.replace(":","__")}.joblib')
        rows=[]
        for c in test:
            Xc=episode_features(frozen,c,mode);base=predict_mode(frozen,c,mode)
            if args.variant=='residual': pred=base+rm.predict(sc.transform(Xc))
            elif args.variant=='calibration': pred=base+alpha*(sc.predict(base)-base)
            else: pred=base+alpha*(sc.predict(base)-base)+beta*Xc[:,0]
            rows.append({'candidate':'M2_meta_residual' if args.variant=='residual' else 'M2_source_calibration_support' if args.variant=='calibration_support' else 'M2_source_calibration','fold':fold,'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,'alpha':alpha,'beta':beta,**metrics(c.y[K:],pred)})
        all_rows.extend(rows);records.append({'fold':fold,'base_mode':mode,'config':config,'chosen':pick,'trials':trials,'meta_source_rows':int(len(y)),'outer_mae':macro(rows),'outer_rmse':macro(rows,'rmse'),'outer_p95_ae':macro(rows,'p95_ae')})
    summary=[]
    for ds in ['XJTU','MATR','Tongji']:
        rows=[r for r in all_rows if r['dataset']==ds]
        name='M2_meta_residual' if args.variant=='residual' else 'M2_source_calibration_support' if args.variant=='calibration_support' else 'M2_source_calibration'
        summary.append({'candidate':name,'dataset':ds,'cells':len(rows),'mae':macro(rows),'rmse':macro(rows,'rmse'),'p95_ae':macro(rows,'p95_ae'),'low_soh_mae':macro(rows,'low_soh_mae')})
    result={'variant':args.variant,'summary':summary,'folds':records,'cell_results':all_rows,'protocol':json.loads((Path(__file__).parent/'protocol.json').read_text())}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'summary':summary,'out':str(out/'summary.json')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
