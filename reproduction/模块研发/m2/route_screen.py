"""Causal source-validation kNN route between SHB and physical GPR views."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import joblib,numpy as np
M1=Path(__file__).resolve().parents[1]/'m1';sys.path.insert(0,str(M1))
from data import load_cells,split,metrics
from features import hi

def macro(rows,key='mae'):
 g=defaultdict(list)
 for r in rows:
  if r.get(key) is not None:g[(r['dataset'],r['domain'])].append(float(r[key]))
 ds=defaultdict(list)
 for (d,_),v in g.items():ds[d].append(np.mean(v))
 return float(np.mean([np.mean(v) for v in ds.values()]))
def mdl(root,name,fold):
 j=root/'jobs'/f'{name}__{fold.replace(":","__")}'
 return joblib.load(j/'model.joblib'),json.loads((j/'tuning.json').read_text())['chosen']['mode']
def pred(m,c,mode):return m.predict(c,mode)
def feat(c):return hi(c.x)[:10].mean(0)
def route(features,labels,k):
 out=[]
 for i in range(len(features)):
  ds=np.sqrt(((features-features[i])**2).sum(1));ds[i]=np.inf
  ids=np.argsort(ds)[:min(k,len(features)-1)]
  if len(ids)==0:out.append(labels[0])
  else:out.append(int(np.mean(labels[ids])>=.5))
 return out
def predict_route(train_features,labels,query,k):
 out=[]
 for q in query:
  ds=np.sqrt(((train_features-q)**2).sum(1));ids=np.argsort(ds)[:min(k,len(ds))]
  out.append(int(np.mean(np.asarray(labels)[ids])>=.5))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='模块研发/results/m2_route_v1');args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 roots={'a':Path('模块研发/results/m1_v1/screen_v2'),'b':Path('模块研发/results/m1_v1/screen_v1')}
 names={'a':'P07_pls_concat','b':'P02_anchor_physics'};cells=[c for d in ['XJTU','MATR','Tongji'] for c in load_cells(d)];folds=sorted({f'{c.dataset}:{c.domain}' for c in cells});all_rows=[];recs=[]
 for fold in folds:
  _,val,test=split(cells,fold);ma,modea=mdl(roots['a'],names['a'],fold);mb,modeb=mdl(roots['b'],names['b'],fold);both=val+test
  pa={c.id:pred(ma,c,modea) for c in both};pb={c.id:pred(mb,c,modeb) for c in both}
  vf=np.array([feat(c) for c in val]);tf=np.array([feat(c) for c in test]);mu=vf.mean(0);sd=np.maximum(vf.std(0),1e-8);vf=(vf-mu)/sd;tf=(tf-mu)/sd
  va=[]
  for c in val:
   ea=metrics(c.y[10:],pa[c.id])['mae'];eb=metrics(c.y[10:],pb[c.id])['mae'];va.append(int(eb<ea))
  trials=[]
  for k in [1,2,3,4,5]:
   choice=route(vf,np.array(va),k);rr=[]
   for c,g in zip(val,choice):rr.append({'dataset':c.dataset,'domain':c.domain,**metrics(c.y[10:],pb[c.id] if g else pa[c.id])})
   trials.append({'k':k,'inner_mae':macro(rr),'inner_rmse':macro(rr,'rmse'),'inner_p95_ae':macro(rr,'p95_ae')})
  pick=min(trials,key=lambda z:(z['inner_mae'],z['inner_rmse'],z['inner_p95_ae']));gates=predict_route(vf,np.array(va),tf,pick['k'])
  rows=[]
  for c,g in zip(test,gates):
   rows.append({'candidate':'M2_source_route','fold':fold,'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,'route':'P02' if g else 'P07',**metrics(c.y[10:],pb[c.id] if g else pa[c.id])})
  all_rows.extend(rows);recs.append({'fold':fold,'chosen':pick,'validation_branch_counts':{'P02':sum(va),'P07':len(va)-sum(va)},'outer_branch_counts':{'P02':sum(gates),'P07':len(gates)},'outer_mae':macro(rows),'outer_rmse':macro(rows,'rmse'),'outer_p95_ae':macro(rows,'p95_ae')})
 summary=[]
 for ds in ['XJTU','MATR','Tongji']:
  rr=[r for r in all_rows if r['dataset']==ds];summary.append({'candidate':'M2_source_route','dataset':ds,'cells':len(rr),'mae':macro(rr),'rmse':macro(rr,'rmse'),'p95_ae':macro(rr,'p95_ae')})
 result={'summary':summary,'folds':recs,'cell_results':all_rows,'protocol':'inner-validation kNN route using first-K HI only'};(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'summary':summary,'out':str(out/'summary.json')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
