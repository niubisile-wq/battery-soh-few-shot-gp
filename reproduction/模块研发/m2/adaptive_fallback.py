"""Three-way, leakage-safe adaptive correction with M1 as a fallback.

For every outer fold, weights of M1, source-calibration/support branch A,
and physical branch B are selected only on validation cells from the same
dataset.  The outer test cells are evaluated once with those frozen weights.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import joblib, numpy as np
M1=Path(__file__).resolve().parents[1]/'m1'; sys.path.insert(0,str(M1))
from data import load_cells, split, metrics  # noqa: E402
from meta_residual import K, episode_features, predict_mode  # noqa: E402

def macro(rows,key='mae'):
    g=defaultdict(list)
    for r in rows:
        v=r.get(key)
        if v is not None:g[(r['dataset'],r['domain'])].append(float(v))
    return float(np.mean([np.mean(v) for v in g.values()]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='模块研发/results/m2_fallback_v1'); args=ap.parse_args()
    r7=Path('模块研发/results/m1_v1/screen_v2'); r2=Path('模块研发/results/m1_v1/screen_v1'); ra=Path('模块研发/results/m2_calibration_support_v3')
    specs=json.loads((M1/'candidates_v1.json').read_text()); s7=specs['P07_pls_concat']
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in load_cells(ds)]
    folds=sorted({f'{c.dataset}:{c.domain}' for c in cells}); all_rows=[]; records=[]
    grid=[(ia/10,ib/10) for ia in range(11) for ib in range(11-ia)]
    for fold in folds:
        train,val,test=split(cells,fold); name=fold.replace(':','__'); ds=fold.split(':',1)[0]
        j7=r7/'jobs'/f'P07_pls_concat__{name}'; m7=joblib.load(j7/'model.joblib'); mode7=json.loads((j7/'tuning.json').read_text())['chosen']['mode']
        info=joblib.load(ra/'fold_models'/f'{name}.joblib'); cal,alpha,beta=info['calibration'],info['alpha'],info['beta']
        def pa(c):
            base=predict_mode(m7,c,mode7); X=episode_features(m7,c,mode7)
            return base+alpha*(cal.predict(base)-base)+beta*X[:,0]
        j2=r2/'jobs'/f'P02_anchor_physics__{name}'; m2=joblib.load(j2/'model.joblib'); mode2=json.loads((j2/'tuning.json').read_text())['chosen']['mode']
        def pb(c):return predict_mode(m2,c,mode2)
        def p0(c):return predict_mode(m7,c,mode7)
        val_ds=[c for c in val if c.dataset==ds]
        # Cache predictions once per cell; weight search must not refit or
        # re-evaluate the expensive GP posterior.
        cached={}
        for c in val_ds+test: cached[c.id]=(p0(c),pa(c),pb(c))
        trials=[]
        for wa,wb in grid:
            rows=[]
            for c in val_ds:
                p0c, pac, pbc=cached[c.id]
                rows.append({'dataset':c.dataset,'domain':c.domain,**metrics(c.y[K:],(1-wa-wb)*p0c+wa*pac+wb*pbc)})
            trials.append({'weight_a':wa,'weight_b':wb,'weight_m1':1-wa-wb,'inner_mae':macro(rows),'inner_rmse':macro(rows,'rmse'),'inner_p95_ae':macro(rows,'p95_ae')})
        pick=min(trials,key=lambda z:(z['inner_mae'],z['inner_rmse'],z['inner_p95_ae']))
        wa,wb=pick['weight_a'],pick['weight_b']; rows=[]
        for c in test:
            p0c,pac,pbc=cached[c.id]; pred=(1-wa-wb)*p0c+wa*pac+wb*pbc
            rows.append({'candidate':'M2_adaptive_fallback','fold':fold,'weight_m1':1-wa-wb,'weight_a':wa,'weight_b':wb,'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,**metrics(c.y[K:],pred)})
        all_rows.extend(rows); records.append({'fold':fold,'chosen':pick,'outer_mae':macro(rows),'outer_rmse':macro(rows,'rmse'),'outer_p95_ae':macro(rows,'p95_ae')})
    summary=[]
    for ds in ['XJTU','MATR','Tongji']:
        rows=[r for r in all_rows if r['dataset']==ds]
        summary.append({'candidate':'M2_adaptive_fallback','dataset':ds,'cells':len(rows),'mae':macro(rows),'rmse':macro(rows,'rmse'),'p95_ae':macro(rows,'p95_ae'),'low_soh_mae':macro(rows,'low_soh_mae')})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps({'summary':summary,'folds':records,'cell_results':all_rows},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summary':summary,'cells':len(all_rows)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
