"""Leakage-safe expert selection for the second module.

The candidates are frozen SHB (M1) with source calibration/support correction
and the already-screened physical-anchor GP.  The branch is selected on the
inner validation cells of each outer fold, then evaluated once on the outer
test cells.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np

M1=Path(__file__).resolve().parents[1]/'m1'; sys.path.insert(0,str(M1))
from data import load_cells, split, source_indices, metrics  # noqa: E402
from gp import GPModel  # noqa: E402
from meta_residual import K, episode_features, predict_mode, prior

def macro(rows, key='mae'):
    g=defaultdict(list)
    for r in rows:
        if key in r and r[key] is not None: g[(r['dataset'],r['domain'])].append(float(r[key]))
    return float(np.mean([np.mean(v) for v in g.values()]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='模块研发/results/m2_expert_v1'); args=ap.parse_args()
    root=Path('模块研发/results/m1_v1/screen_v2'); p02root=Path('模块研发/results/m1_v1/screen_v1')
    m2root=Path('模块研发/results/m2_calibration_support_v3')
    specs=json.loads((M1/'candidates_v1.json').read_text())
    spec7=specs['P07_pls_concat']; spec2=specs['P02_anchor_physics']
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in load_cells(ds)]
    folds=sorted({f'{c.dataset}:{c.domain}' for c in cells}); all_rows=[]; records=[]
    for fold in folds:
        train,val,test=split(cells,fold)
        name=fold.replace(':','__')
        j7=root/'jobs'/f'P07_pls_concat__{name}'; m7=joblib.load(j7/'model.joblib')
        mode7=json.loads((j7/'tuning.json').read_text())['chosen']['mode']
        ainfo=joblib.load(m2root/'fold_models'/f'{name}.joblib')
        cal,alpha,beta=ainfo['calibration'],ainfo['alpha'],ainfo['beta']
        def pred_a(c, model=m7, mode=mode7):
            base=predict_mode(model,c,mode); X=episode_features(model,c,mode)
            return base+alpha*(cal.predict(base)-base)+beta*X[:,0]
        j2=p02root/'jobs'/f'P02_anchor_physics__{name}'; m2=joblib.load(j2/'model.joblib')
        mode2=json.loads((j2/'tuning.json').read_text())['chosen']['mode']
        def pred_b(c, model=m2, mode=mode2): return predict_mode(model,c,mode)
        val_a=[]; val_b=[]
        for c in val:
            val_a.append({'branch':'A','dataset':c.dataset,'domain':c.domain,'cell_id':c.id,**metrics(c.y[K:],pred_a(c))})
            val_b.append({'branch':'B','dataset':c.dataset,'domain':c.domain,'cell_id':c.id,**metrics(c.y[K:],pred_b(c))})
        target_ds=fold.split(':',1)[0]
        val_sel=[c for c in val if c.dataset==target_ds]
        trials=[]
        for w in np.linspace(0,1,11):
            rows=[]
            for c in val_sel:
                pa,pb=pred_a(c),pred_b(c)
                rows.append({'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,**metrics(c.y[K:],w*pa+(1-w)*pb)})
            trials.append({'weight_a':float(w),'inner_mae':macro(rows),'inner_rmse':macro(rows,'rmse'),'inner_p95_ae':macro(rows,'p95_ae')})
        pick=min(trials,key=lambda z:(z['inner_mae'],z['inner_rmse'],z['inner_p95_ae']))
        weight_a=pick['weight_a']
        rows=[]
        for c in test:
            pred=weight_a*pred_a(c)+(1-weight_a)*pred_b(c)
            rows.append({'candidate':'M2_adaptive_blend','fold':fold,'weight_a':weight_a,'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,**metrics(c.y[K:],pred)})
        all_rows.extend(rows)
        records.append({'fold':fold,'weight_a':weight_a,'inner_trials':trials,'outer_mae':macro(rows),'outer_rmse':macro(rows,'rmse'),'outer_p95_ae':macro(rows,'p95_ae')})
    summary=[]
    for ds in ['XJTU','MATR','Tongji']:
        rows=[r for r in all_rows if r['dataset']==ds]
        summary.append({'candidate':'M2_adaptive_blend','dataset':ds,'cells':len(rows),'mae':macro(rows),'rmse':macro(rows,'rmse'),'p95_ae':macro(rows,'p95_ae'),'low_soh_mae':macro(rows,'low_soh_mae')})
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps({'summary':summary,'folds':records,'cell_results':all_rows},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summary':summary,'weights':{str(float(w)):int(sum(abs(r['weight_a']-w)<1e-9 for r in records)) for w in np.linspace(0,1,11)},'cells':len(all_rows)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
