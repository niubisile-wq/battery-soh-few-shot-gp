"""INVALID historical experiment: pooled validation leaks across outer folds.

Retained for provenance. Use strict_ablation.py for valid fold-local evaluation.
"""
import json,glob
from collections import defaultdict
from pathlib import Path
import numpy as np,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'m1'))
from data import metrics

def macro(rs,k):
 g=defaultdict(list)
 for r in rs:
  if r.get(k) is not None:g[(r['dataset'],r['domain'])].append(float(r[k]))
 return float(np.mean([np.mean(v) for v in g.values()]))

def make_cell(rows,w):
 out=[]
 for cid in sorted({r['cell_id'] for r in rows}):
  rr=[r for r in rows if r['cell_id']==cid];y=np.asarray([r['y'] for r in rr]);p=np.asarray([w*r['pa']+(1-w)*r['pb'] for r in rr]);base=metrics(y,p);out.append({'fold':rr[0]['fold'],'dataset':rr[0]['dataset'],'domain':rr[0]['domain'],'cell_id':cid,'weight_a':float(w),**base})
 return out

def main(out='模块研发/results/m2_global_domain_blend_v1'):
 raise RuntimeError('INVALID cross-fold validation pooling; use strict_ablation.py. Historical results are withdrawn.')

def historical_invalid_implementation(out):
 raw=[]
 for p in glob.glob('模块研发/results/m2_prediction_cache_v1/*.json'):
  for r in json.loads(Path(p).read_text()): r['fold']=Path(p).stem; raw.append(r)
 weights={};selection={}
 for d in ['XJTU','MATR','Tongji']:
  vr=[r for r in raw if r['split']=='val' and r['dataset']==d];trials=[]
  for w in np.linspace(0,1,101):
   cr=make_cell(vr,w);trials.append({'weight_a':float(w),'mae':macro(cr,'mae'),'rmse':macro(cr,'rmse'),'p95_ae':macro(cr,'p95_ae')})
  pick=min(trials,key=lambda x:(x['mae'],x['rmse'],x['p95_ae']));weights[d]=pick['weight_a'];selection[d]={'chosen':pick,'trials':trials}
 test=[]
 for d,w in weights.items():test.extend(make_cell([r for r in raw if r['split']=='test' and r['dataset']==d],w))
 summary=[]
 for d in ['XJTU','MATR','Tongji']:
  rs=[r for r in test if r['dataset']==d];summary.append({'candidate':'M2_global_domain_blend','dataset':d,'cells':len(rs),'mae':macro(rs,'mae'),'rmse':macro(rs,'rmse'),'p95_ae':macro(rs,'p95_ae'),'low_soh_mae':macro(rs,'low_soh_mae')})
 p=Path(out);p.mkdir(parents=True,exist_ok=True);(p/'summary.json').write_text(json.dumps({'candidate':'M2_global_domain_blend','weights':weights,'selection':selection,'summary':summary,'cell_results':test,'protocol':{'selection':'dataset-level weight chosen from validation rows only','test_future_labels_used':False}},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'weights':weights,'summary':summary,'cells':len(test)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
