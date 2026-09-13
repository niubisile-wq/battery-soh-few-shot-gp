"""Cheap, leakage-safe query-level gate search over cached M1/A/B outputs."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'m1'))
from data import metrics

def macro(rows,key='mae'):
    g=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:g[(r['dataset'],r['domain'])].append(float(r[key]))
    return float(np.mean([np.mean(v) for v in g.values()]))

def pred(rule,r):
    f=r[rule['feature']]
    if rule['kind']=='threshold': use=f>=rule['threshold']
    elif rule['kind']=='abs_threshold': use=abs(f)>=rule['threshold']
    else: raise ValueError(rule)
    pa,p0,pb=r['pa'],r['p0'],r['pb']; chosen=pa if rule['high']=='a' else (pb if rule['high']=='b' else p0); other=pa if rule['low']=='a' else (pb if rule['low']=='b' else p0)
    return chosen if use else other

def main(out='模块研发/results/m2_gate_v1'):
    cache=Path('模块研发/results/m2_prediction_cache_v1'); files=sorted(cache.glob('*.json')); all_rows=[]; records=[]
    features=['resid','dis_ab','dis_a0','dis_b0']; rules=[]
    for feature in features:
        for kind in ['threshold','abs_threshold']:
            for high in ['a','b','0']:
                for low in ['a','b','0']:
                    if high==low:continue
                    rules.append((feature,kind,high,low))
    for fp in files:
        rows=json.loads(fp.read_text()); ds=rows[0]['dataset'];val=[r for r in rows if r['split']=='val'];test=[r for r in rows if r['split']=='test'];trials=[]
        y=np.asarray([r['y'] for r in val]); p0=np.asarray([r['p0'] for r in val]); pa=np.asarray([r['pa'] for r in val]); pb=np.asarray([r['pb'] for r in val]);
        keys=[(r['dataset'],r['domain']) for r in val]; groups=sorted(set(keys));
        for feature,kind,high,low in rules:
            vals=np.asarray([abs(r[feature]) if kind=='abs_threshold' else r[feature] for r in val],float); qs=np.unique(np.quantile(vals,np.linspace(.1,.9,9)))
            for t in qs:
                rule={'feature':feature,'kind':kind,'high':high,'low':low,'threshold':float(t)}
                f=vals>=t; a=pa if high=='a' else (pb if high=='b' else p0); b=pa if low=='a' else (pb if low=='b' else p0); pp=np.where(f,a,b); maes=[]; rms=[]
                for g in groups:
                    ix=np.asarray([k==g for k in keys]); e=pp[ix]-y[ix];maes.append(np.abs(e).mean());rms.append(np.sqrt(np.mean(e**2)))
                trials.append((float(np.mean(maes)),float(np.mean(rms)),rule))
        pick=min(trials,key=lambda x:(x[0],x[1]))[2];outrows=[]
        for r in test:
            outrows.append({'candidate':'M2_query_gate','fold':fp.stem,'rule':pick,'dataset':r['dataset'],'domain':r['domain'],'cell_id':r['cell_id'],'q':r['q'],**metrics(np.asarray([r['y']]),np.asarray([pred(pick,r)]))})
        cellrows=[]
        for cid in sorted({r['cell_id'] for r in outrows}):
            raw=[r for r in test if r['cell_id']==cid]
            yp=np.asarray([pred(pick,r) for r in raw]); yy=np.asarray([r['y'] for r in raw])
            cellrows.append({'candidate':'M2_query_gate','fold':fp.stem,'rule':pick,'dataset':raw[0]['dataset'],'domain':raw[0]['domain'],'cell_id':cid,**metrics(yy,yp)})
        best_trial=min(trials,key=lambda x:(x[0],x[1])); all_rows.extend(cellrows); records.append({'fold':fp.stem,'rule':pick,'inner_mae':best_trial[0],'outer_mae':macro(cellrows)})
    summary=[]
    for ds in ['XJTU','MATR','Tongji']:
        rs=[r for r in all_rows if r['dataset']==ds];summary.append({'candidate':'M2_query_gate','dataset':ds,'cells':len(rs),'mae':macro(rs),'rmse':macro(rs,'rmse'),'p95_ae':macro(rs,'p95_ae'),'low_soh_mae':macro(rs,'low_soh_mae')})
    p=Path(out);p.mkdir(parents=True,exist_ok=True);(p/'summary.json').write_text(json.dumps({'summary':summary,'folds':records,'cell_results':all_rows},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'summary':summary,'cells':len(all_rows)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
