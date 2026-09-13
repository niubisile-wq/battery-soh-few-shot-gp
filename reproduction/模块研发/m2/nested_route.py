"""Nested, training-only route selection for the A/B second-module branches."""
import json,sys
from collections import defaultdict
from pathlib import Path
import joblib,numpy as np
M1=Path(__file__).resolve().parents[1]/'m1';sys.path.insert(0,str(M1))
from data import load_cells,split,source_indices,metrics  # noqa:E402
from gp import GPModel  # noqa:E402
from meta_residual import K,episode_features,predict_mode,prior  # noqa:E402

def macro(rs,k='mae'):
 g=defaultdict(list)
 for r in rs:
  if r.get(k) is not None:g[(r['dataset'],r['domain'])].append(r[k])
 return float(np.mean([np.mean(v) for v in g.values()])) if g else 1e9

def main(out='模块研发/results/m2_nested_route_v1'):
 r7=Path('模块研发/results/m1_v1/screen_v2');r2=Path('模块研发/results/m1_v1/screen_v1');ra=Path('模块研发/results/m2_calibration_support_v3');specs=json.loads((M1/'candidates_v1.json').read_text());s7=specs['P07_pls_concat'];s2=specs['P02_anchor_physics'];cells=[c for d in ['XJTU','MATR','Tongji'] for c in load_cells(d)];folds=sorted({f'{c.dataset}:{c.domain}' for c in cells});allrows=[];recs=[]
 for fold in folds:
  train,val,test=split(cells,fold);name=fold.replace(':','__');j7=r7/'jobs'/f'P07_pls_concat__{name}';j2=r2/'jobs'/f'P02_anchor_physics__{name}';mode7=json.loads((j7/'tuning.json').read_text())['chosen']['mode'];cfg7=json.loads((j7/'tuning.json').read_text())['chosen']['config'];mode2=json.loads((j2/'tuning.json').read_text())['chosen']['mode'];cfg2=json.loads((j2/'tuning.json').read_text())['chosen']['config'];inner=[]; ordered=sorted(train,key=lambda c:c.id)
  for held in [ordered[::2],ordered[1::2]]:
   ids={c.id for c in held};fit=[c for c in train if c.id not in ids];idx=source_indices(fit);ma=GPModel(s7,cfg7).fit(fit,idx);mb=GPModel(s2,cfg2).fit(fit,idx)
   for c in held:
    p0=predict_mode(ma,c,mode7);xa=episode_features(ma,c,mode7);pa=p0+xa[:,0];pb=predict_mode(mb,c,mode2)
    for branch,p in [('A',pa),('B',pb)]:inner.append({'branch':branch,'dataset':c.dataset,'domain':c.domain,**metrics(c.y[K:],p)})
  ds=fold.split(':',1)[0]; choices={}
  for d in ['XJTU','MATR','Tongji']:
   aa=[r for r in inner if r['dataset']==d and r['branch']=='A'];bb=[r for r in inner if r['dataset']==d and r['branch']=='B'];choices[d]='A' if macro(aa)<macro(bb) else 'B'
  m7=joblib.load(j7/'model.joblib'); info=joblib.load(ra/'fold_models'/f'{name}.joblib');cal,alpha,beta=info['calibration'],info['alpha'],info['beta'];m2=joblib.load(j2/'model.joblib')
  rows=[]
  for c in test:
   p0=predict_mode(m7,c,mode7);x=episode_features(m7,c,mode7);pa=p0+alpha*(cal.predict(p0)-p0)+beta*x[:,0];pb=predict_mode(m2,c,mode2);p=pa if choices[c.dataset]=='A' else pb;rows.append({'candidate':'M2_nested_route','fold':fold,'route':choices[c.dataset],'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,**metrics(c.y[K:],p)})
  allrows.extend(rows);recs.append({'fold':fold,'choices':choices,'inner':{d:{'A':macro([r for r in inner if r['dataset']==d and r['branch']=='A']),'B':macro([r for r in inner if r['dataset']==d and r['branch']=='B'])} for d in choices}})
 summary=[]
 for d in ['XJTU','MATR','Tongji']:
  rs=[r for r in allrows if r['dataset']==d];summary.append({'candidate':'M2_nested_route','dataset':d,'cells':len(rs),'mae':macro(rs),'rmse':macro(rs,'rmse'),'p95_ae':macro(rs,'p95_ae'),'low_soh_mae':macro(rs,'low_soh_mae')})
 p=Path(out);p.mkdir(parents=True,exist_ok=True);(p/'summary.json').write_text(json.dumps({'summary':summary,'folds':recs,'cell_results':allrows},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'summary':summary,'cells':len(allrows)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
