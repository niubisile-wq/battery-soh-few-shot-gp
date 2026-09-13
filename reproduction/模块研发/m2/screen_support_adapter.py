"""Screen a second module on top of the frozen first-module SHB fits."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np

M1 = Path(__file__).resolve().parents[1] / 'm1'
sys.path.insert(0, str(M1))
from data import load_cells, split, metrics  # noqa: E402

K = 10
MODES = ['source_only', 'source_bias', 'posterior']
LAMBDAS = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]

def macro(rows, key='mae'):
    grouped=defaultdict(list)
    for r in rows:
        if r.get(key) is not None: grouped[(r['dataset'],r['domain'])].append(float(r[key]))
    ds=defaultdict(list)
    for (d,_), vals in grouped.items(): ds[d].append(float(np.mean(vals)))
    return float(np.mean([np.mean(v) for v in ds.values()]))

def load_model(root, fold):
    job=root/'jobs'/f'P07_pls_concat__{fold.replace(":","__")}'
    return joblib.load(job/'model.joblib'), json.loads((job/'tuning.json').read_text())['chosen']

def full_mean(model, cell):
    z=model.features.transform(cell)
    p=model.gp.predict(z)
    if model.mean_model is not None: p=p+model.mean_model.predict(z)
    return p+0.0

def predict_mode(model, cell, mode):
    if mode == 'posterior': return model.predict(cell, mode)
    p=full_mean(model, cell)
    if mode == 'source_bias': return p[K:]+(cell.y[:K]-p[:K]).mean()
    return p[K:]

def cell_row(cell, pred, candidate, fold, mode, lam):
    return {'candidate':candidate,'fold':fold,'dataset':cell.dataset,'domain':cell.domain,
            'cell_id':cell.id,'mode':mode,'lambda':lam,**metrics(cell.y[K:],pred)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='模块研发/results/m1_v1/screen_v2')
    ap.add_argument('--out',default='模块研发/results/m2_v1')
    args=ap.parse_args(); root=Path(args.root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    cells=[c for d in ['XJTU','MATR','Tongji'] for c in load_cells(d)]
    folds=sorted({f'{c.dataset}:{c.domain}' for c in cells}); all_rows=[]; fold_records=[]
    for fold in folds:
        _, val, test=split(cells,fold); model, original=load_model(root,fold)
        cached={c.id:{'full':full_mean(model,c),'res':c.y[:K]-full_mean(model,c)[:K]} for c in val+test}
        pred_cache={c.id:{mode:predict_mode(model,c,mode) for mode in MODES} for c in val+test}
        trials=[]
        for mode in MODES:
            for lam in LAMBDAS:
                rows=[]
                for c in val:
                    base=pred_cache[c.id][mode]
                    pred=base+lam*cached[c.id]['res'].mean()
                    rows.append(cell_row(c,pred,'M2_trial',fold,mode,lam))
                trials.append({'mode':mode,'lambda':lam,'inner_mae':macro(rows),
                               'inner_rmse':macro(rows,'rmse'),'inner_p95_ae':macro(rows,'p95_ae')})
        chosen=min(trials,key=lambda r:(r['inner_mae'],r['inner_rmse'],r['inner_p95_ae']))
        rows=[]
        for c in test:
            base=pred_cache[c.id][chosen['mode']]
            pred=base+chosen['lambda']*cached[c.id]['res'].mean()
            rows.append(cell_row(c,pred,'M2_support_adapter',fold,chosen['mode'],chosen['lambda']))
        all_rows.extend(rows)
        fold_records.append({'fold':fold,'original_shb_choice':original,'chosen':chosen,
                             'inner_trials':trials,'outer_mae':macro(rows),
                             'outer_rmse':macro(rows,'rmse'),'outer_p95_ae':macro(rows,'p95_ae')})
    summary=[]
    for ds in ['XJTU','MATR','Tongji']:
        rows=[r for r in all_rows if r['dataset']==ds]
        summary.append({'candidate':'M2_support_adapter','dataset':ds,'cells':len(rows),
                        'mae':macro(rows),'rmse':macro(rows,'rmse'),'p95_ae':macro(rows,'p95_ae'),
                        'low_soh_mae':macro(rows,'low_soh_mae')})
    result={'protocol':json.loads((Path(__file__).parent/'protocol.json').read_text()),
            'summary':summary,'folds':fold_records,'cell_results':all_rows}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summary':summary,'out':str(out/'summary.json')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
