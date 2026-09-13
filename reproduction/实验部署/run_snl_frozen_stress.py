from pathlib import Path
import sys,zipfile,pickle,csv,json,time,hashlib
import numpy as np, joblib
from sklearn.metrics import mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'开发基线选择依据'),str(ROOT/'模块研发/m1'),str(ROOT/'模块研发/m2'),str(ROOT/'模块研发/m3'),str(ROOT/'模块研发/results/m4_random_function_candidate_v1/source_snapshot/m4'),str(ROOT/'模块研发/results/m4_random_function_candidate_v1/source_snapshot/m3'),str(ROOT/'模块研发/results/m4_random_function_candidate_v1/source_snapshot/m2'),str(ROOT/'模块研发/results/m4_random_function_candidate_v1/source_snapshot/m1')]
from data import Cell
from partial_charge_protocol import Window,extract
OUT=ROOT/'外部泛化方案二审计_20260913'; ZIP=ROOT/'数据集/BatteryLife_v12_processed/SNL.zip'; RELEASE=ROOT/'模块研发/results/m4_random_function_candidate_v1/folds'
FOLDS=sorted(RELEASE.glob('XJTU__*/B1234/adapter.joblib'))
def metrics(y,p):
 e=(np.asarray(p)-np.asarray(y))*100; a=np.abs(e)
 return dict(n=len(e),mae=float(a.mean()),rmse=float(np.sqrt(np.mean(e**2))),p95=float(np.quantile(a,.95)),bias=float(e.mean()))
def load_cell(o,n):
 cut=float(o['max_voltage_limit_in_V']);nom=float(o['nominal_capacity_in_Ah']); xs=[];caps=[];nums=[]
 for c in sorted(o['cycle_data'],key=lambda c:float(c['cycle_number'])):
  try:
   i=np.asarray(c['current_in_A'],float);v=np.asarray(c['voltage_in_V'],float);q=np.asarray(c['discharge_capacity_in_Ah'],float);dis=i<-.05*nom
   if not dis.any():continue
   cap=float(q[dis].max());x,m=extract(c,Window(cut-.3,cut-.1));xs.append(x);caps.append(cap);nums.append(m['cycle_number'])
  except Exception:continue
 if len(xs)<11:return None
 q=np.asarray(caps,float); return Cell(n,'SNL',n,np.asarray(xs,np.float32),q/q[0],np.asarray(nums,np.float32),float(q[0]),nom,n)
def main():
 rows=[];cells=0; eligible=0; t0=time.time()
 with zipfile.ZipFile(ZIP) as z:
  names=[n for n in z.namelist() if n.endswith('.pkl')]
  for ix,n in enumerate(names,1):
   c=load_cell(pickle.loads(z.read(n)),n)
   if c is None:continue
   cells+=1;eligible+=len(c.y)-10
   for fp in FOLDS:
    wrapper=joblib.load(fp)
    base=wrapper.parent.parent.parent # B1234 -> M3 -> M2 -> GP; actual parent chain
    while hasattr(base,'parent'): base=base.parent
    pB=base.predict(c, "source_bias")
    pFull=wrapper.predict(c)
    y=c.y[10:]
    for model,p in [('B',pB),('B1234',pFull)]:
     m=metrics(y,p); rows.append(dict(dataset='SNL',cell_id=n,fold=fp.parts[-3],model=model,query_count=len(y),**m))
   print(ix,len(names),n,'eligible',len(c.y),'elapsed',round(time.time()-t0,1),flush=True)
 with (OUT/'SNL冻结模型压力测试_逐轨迹.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 agg=[]
 for model in ['B','B1234']:
  rr=[r for r in rows if r['model']==model]
  agg.append(dict(dataset='SNL',model=model,trajectories=len(rr),cells=len(set(r['cell_id'] for r in rr)),query_points=sum(r['n'] for r in rr),mae=float(np.mean([r['mae'] for r in rr])),rmse=float(np.mean([r['rmse'] for r in rr])),p95=float(np.mean([r['p95'] for r in rr])),bias=float(np.mean([r['bias'] for r in rr]))))
 (OUT/'SNL冻结模型压力测试_汇总.json').write_text(json.dumps({'status':'COMPLETE_HISTORICALLY_EXPOSED_STRESS_TEST','dataset':'SNL','cells':cells,'eligible_query_points':eligible,'folds':[str(x.relative_to(ROOT)) for x in FOLDS],'aggregation':'equal mean over cell-fold trajectories; six source-fold models, no new ensemble','results':agg,'independence':'FAIL_SNL_HISTORICALLY_SCORED_IN_MULTIDATASET_HI_MATRIX','zip_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest()},indent=2),encoding='utf8')
 print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
