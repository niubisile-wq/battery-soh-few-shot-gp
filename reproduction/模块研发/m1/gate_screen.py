"""Inner-validation-only disagreement gate for P07 and P02 GPR branches."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from data import load_cells, split, metrics

def macro(rows, key='mae'):
    g=defaultdict(list)
    for r in rows:g[(r['dataset'],r['domain'])].append(float(r[key]))
    ds=defaultdict(list)
    for (d,_),v in g.items():ds[d].append(np.mean(v))
    return float(np.mean([np.mean(v) for v in ds.values()]))

def chosen_mode(root,name,fold):
    p=root/'jobs'/f'{name}__{fold.replace(":","__")}'/'tuning.json'
    return json.loads(p.read_text())['chosen']['mode']

def model(root,name,fold):
    return joblib.load(root/'jobs'/f'{name}__{fold.replace(":","__")}'/'model.joblib')

def rows(cells, pa, pb, gate):
    out=[]
    for c in cells:
        p=pa[c.id] if gate[c.id] == 'a' else pb[c.id]
        out.append({'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,**metrics(c.y[10:],p)})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root-a',default='模块研发/results/m1_v1/screen_v2')
    ap.add_argument('--root-b',default='模块研发/results/m1_v1/screen_v1')
    ap.add_argument('--name-a',default='P07_pls_concat')
    ap.add_argument('--name-b',default='P02_anchor_physics')
    ap.add_argument('--out',default='模块研发/results/m1_v1/gate_p07_p02_v1.json')
    args=ap.parse_args(); ra=Path(args.root_a); rb=Path(args.root_b)
    datasets=['XJTU','MATR','Tongji']; cells=[c for d in datasets for c in load_cells(d)]
    folds=sorted({f'{c.dataset}:{c.domain}' for c in cells}); results=[]
    for fold in folds:
        _,va,te=split(cells,fold); ma=model(ra,args.name_a,fold); mb=model(rb,args.name_b,fold)
        modea=chosen_mode(ra,args.name_a,fold); modeb=chosen_mode(rb,args.name_b,fold)
        both=va+te
        pa={c.id:ma.predict(c,modea) for c in both}; pb={c.id:mb.predict(c,modeb) for c in both}
        disagreement={c.id:float(np.mean(abs(pa[c.id]-pb[c.id]))) for c in both}
        thresholds=np.unique(np.quantile([disagreement[c.id] for c in va],np.linspace(0,1,21)))
        trials=[]
        for direction in ['low_b','high_b']:
            for threshold in thresholds:
                gate={c.id:('b' if ((disagreement[c.id]<=threshold) if direction=='low_b' else (disagreement[c.id]>=threshold)) else 'a') for c in va}
                rr=rows(va,pa,pb,gate); trials.append({'direction':direction,'threshold':float(threshold),'inner_mae':macro(rr),'inner_rmse':macro(rr,'rmse'),'inner_p95_ae':macro(rr,'p95_ae')})
        trials += [{'direction':'all_a','threshold':None,'inner_mae':macro(rows(va,pa,pb,{c.id:'a' for c in va})), 'inner_rmse':macro(rows(va,pa,pb,{c.id:'a' for c in va}),'rmse'),'inner_p95_ae':macro(rows(va,pa,pb,{c.id:'a' for c in va}),'p95_ae')},
                   {'direction':'all_b','threshold':None,'inner_mae':macro(rows(va,pa,pb,{c.id:'b' for c in va})), 'inner_rmse':macro(rows(va,pa,pb,{c.id:'b' for c in va}),'rmse'),'inner_p95_ae':macro(rows(va,pa,pb,{c.id:'b' for c in va}),'p95_ae')}]
        ch=min(trials,key=lambda x:(x['inner_mae'],x['inner_rmse'],x['inner_p95_ae']))
        gate={c.id:('a' if ch['direction']=='all_a' else 'b' if ch['direction']=='all_b' else ('b' if ((disagreement[c.id]<=ch['threshold']) if ch['direction']=='low_b' else (disagreement[c.id]>=ch['threshold'])) else 'a')) for c in te}
        rr=rows(te,pa,pb,gate); results.append({'fold':fold,'chosen':ch,'mode_a':modea,'mode_b':modeb,'mae':macro(rr),'rmse':macro(rr,'rmse'),'p95_ae':macro(rr,'p95_ae'),'rows':rr})
    summary=[]
    for ds in datasets:
        rr=[x for r in results if r['fold'].startswith(ds+':') for x in r['rows']]
        summary.append({'dataset':ds,'candidate':'disagreement_gate_P07_P02','cells':len(rr),'mae':macro(rr),'rmse':macro(rr,'rmse'),'p95_ae':macro(rr,'p95_ae')})
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'summary':summary,'folds':results},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'summary':summary,'out':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
