from pathlib import Path
import sys,zipfile,pickle,csv,json,time,hashlib
import numpy as np,joblib
from sklearn.metrics import mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'外部泛化方案二审计_20260913';BASE=ROOT/'模块研发/results/m4_random_function_candidate_v1/folds'
sys.path[:0]=[str(ROOT/'开发基线选择依据'),str(ROOT/'模块研发/m1'),str(ROOT/'模块研发/m2'),str(ROOT/'模块研发/m3'),str(ROOT/'模块研发/results/m4_random_function_candidate_v1/source_snapshot/m4'),str(ROOT/'模块研发/results/m4_random_function_candidate_v1/source_snapshot/m3'),str(ROOT/'模块研发/results/m4_random_function_candidate_v1/source_snapshot/m2'),str(ROOT/'模块研发/results/m4_random_function_candidate_v1/source_snapshot/m1')]
from data import Cell
from partial_charge_protocol import Window,extract
FOLDS=sorted(BASE.glob('XJTU__*/B1234/adapter.joblib'))
ARCH={'UL_PUR':ROOT/'数据集/外部候选预注册_20260913/UL_PUR.zip','CALB':ROOT/'数据集/外部候选预注册_20260913/CALB.zip','MICH':ROOT/'数据集/外部候选预注册_20260913/MICH.zip'}
def met(y,p):
 e=(np.asarray(p)-np.asarray(y))*100;a=np.abs(e);return len(e),float(a.mean()),float(np.sqrt(np.mean(e**2))),float(np.quantile(a,.95)),float(e.mean())
def load(o,n):
 cut=float(o['max_voltage_limit_in_V']);nom=float(o['nominal_capacity_in_Ah']);xs=[];q=[];nums=[]
 for c in sorted(o['cycle_data'],key=lambda z:float(z['cycle_number'])):
  try:
   i=np.asarray(c['current_in_A']);v=np.asarray(c['voltage_in_V']);cap=np.asarray(c['discharge_capacity_in_Ah']);dis=i<-.05*nom
   if not dis.any():continue
   qq=float(cap[dis].max());x,m=extract(c,Window(cut-.3,cut-.1));xs.append(x);q.append(qq);nums.append(m['cycle_number'])
  except Exception:continue
 if len(xs)<11:return None
 q=np.asarray(q);return Cell(n,'EXT',n,np.asarray(xs,np.float32),q/q[0],np.asarray(nums,np.float32),float(q[0]),nom,n)
def main():
 rows=[];start=time.time()
 for ds,archive in ARCH.items():
  with zipfile.ZipFile(archive) as z:
   names=[n for n in z.namelist() if n.endswith('.pkl')]
   for ix,n in enumerate(names,1):
    c=load(pickle.loads(z.read(n)),n)
    if c is None: print(ds,n,'EXCLUDED_PROTOCOL',flush=True);continue
    for fp in FOLDS:
     w=joblib.load(fp);b=w.parent.parent.parent
     while hasattr(b,'parent'): b=b.parent
     pb=b.predict(c,'source_bias');pf=w.predict(c);y=c.y[10:]
     for model,p in [('B',pb),('B1234',pf)]:
      npt,mae,rmse,p95,bias=met(y,p);rows.append(dict(dataset=ds,cell_id=n,protocol=n.rsplit('/',1)[-1].replace('.pkl',''),fold=fp.parts[-3],model=model,query_count=npt,mae=mae,rmse=rmse,p95=p95,bias=bias))
    print(ds,ix,len(names),n,'eligible',len(c.y),flush=True)
 with (OUT/'预注册外部测试_逐轨迹.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 summary=[]
 for ds in ARCH:
  for model in ['B','B1234']:
   rr=[r for r in rows if r['dataset']==ds and r['model']==model];summary.append(dict(dataset=ds,model=model,cells=len(set(r['cell_id'] for r in rr)),trajectories=len(rr),query_points=sum(int(r['query_count']) for r in rr),mae=float(np.mean([r['mae'] for r in rr])),rmse=float(np.mean([r['rmse'] for r in rr])),p95=float(np.mean([r['p95'] for r in rr])),bias=float(np.mean([r['bias'] for r in rr]))))
 # pooled equal trajectory and paired cell counts
 paired={}
 for r in rows: paired.setdefault((r['dataset'],r['cell_id']),{})[r['model']]=r['mae']
 for ds in ARCH:
  dif=[v['B1234']-v['B'] for (d,c),v in paired.items() if d==ds and set(v)=={'B','B1234'}]
  b=np.random.default_rng(0).choice(np.asarray(dif),size=(20000,len(dif)),replace=True).mean(1)
  summary.append(dict(dataset=ds,model='paired_change',cells=len(dif),improved_cells=int(sum(x<0 for x in dif)),worse_cells=int(sum(x>0 for x in dif)),change_pp=float(np.mean(dif)),change_pct=float(np.mean(dif)/np.mean([v['B'] for (d,c),v in paired.items() if d==ds])*100),bootstrap95=[float(np.quantile(b,.025)),float(np.quantile(b,.975))]))
 allrr=[r for r in rows if r['model']=='B'];full=[r for r in rows if r['model']=='B1234'];
 summary.append(dict(dataset='ALL_PREREGISTERED',model='pooled',cells=len(set(r['cell_id'] for r in allrr)),trajectories=len(allrr),query_points=sum(int(r['query_count']) for r in allrr),mae=float(np.mean([r['mae'] for r in allrr])),rmse=float(np.mean([r['rmse'] for r in allrr])),p95=float(np.mean([r['p95'] for r in allrr])),b1234_mae=float(np.mean([r['mae'] for r in full])),change_pct=float((np.mean([r['mae'] for r in full])/np.mean([r['mae'] for r in allrr])-1)*100)))
 out={'status':'COMPLETE_PREREGISTERED_EXTERNAL_TEST','registered_protocol':str((OUT/'外部候选预注册.json').relative_to(ROOT)),'datasets':list(ARCH),'folds':6,'summary':summary,'archive_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in ARCH.items()}}
 (OUT/'预注册外部测试_汇总.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
