#!/usr/bin/env python3
"""Use Oxford/NASA in their valid role: sparse RPT-only trajectory checks."""
import csv,io,json,zipfile
from pathlib import Path
import numpy as np
import scipy.io as sio
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'results'/'legacy_rpt_external';K=10
def normalize(y):
 r=float(np.mean(y[:3]));return y/r if r>0 else y
def metric(y,p):return float(mean_absolute_error(y,p)),float(mean_squared_error(y,p)**.5)
def oxford():
 p=ROOT/'数据集/Oxford_Battery_Degradation_Dataset_1/Oxford_Battery_Degradation_Dataset_1.mat';z=sio.loadmat(p,squeeze_me=True,struct_as_record=False);out=[]
 for name in sorted(k for k in z if k.startswith('Cell')):
  c=z[name];ys=[]
  for cyc in c._fieldnames:
   o=getattr(c,cyc);d=getattr(o,'C1dc');q=np.asarray(d.q,dtype=float);ys.append(float(np.nanmax(q)))
  y=normalize(np.asarray(ys,dtype='float32'));out.append((name,y))
 return out
def nasa():
 p=ROOT/'数据集/NASA/NASA_PCoE_Battery_Data_Set.zip';out=[]
 with zipfile.ZipFile(p) as outer:
  for nested in [n for n in outer.namelist() if n.endswith('.zip')]:
   with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as inner:
    for fn in inner.namelist():
     if not fn.endswith('.mat'):continue
     raw=sio.loadmat(io.BytesIO(inner.read(fn)),squeeze_me=True,struct_as_record=False);name=fn.rsplit('/',1)[-1].replace('.mat','');b=raw.get(name)
     if b is None:continue
     ys=[]
     for c in np.atleast_1d(b.cycle):
      if getattr(c,'type','')=='discharge':
       cap=getattr(c.data,'Capacity',None)
       if cap is not None:
        vals=np.ravel(np.asarray(cap,dtype=float))
        if vals.size:ys.append(float(vals[-1]))
     if len(ys)>K:out.append((name,normalize(np.asarray(ys,dtype='float32'))))
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];cells=[]
 for dataset,data in [('Oxford',oxford()),('NASA',nasa())]:
  for cid,y in data:
   if len(y)<=K:continue
   q=y[K:];preds={'last_support':np.full(len(q),float(y[K-1])),'linear_trend':formal.linear_trend(None,y[:K],len(q),float(y.mean()))}
   for model,p in preds.items():
    a,b=metric(q,p);rows.append({'dataset':dataset,'model':model,'variant':'legacy_RPT_only','K':K,'cell_id':cid,'query_count':len(q),'mae':a,'rmse':b});cells.append(rows[-1])
 with (OUT/'legacy_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 (OUT/'manifest.json').write_text(json.dumps({'datasets':['Oxford','NASA'],'role':'sparse RPT-only external trajectory check','raw_curve_cross_dataset_ranking':False,'K':K,'cells':len(set((x['dataset'],x['cell_id']) for x in rows))},indent=2),encoding='utf8');print(json.dumps({'rows':len(rows)},indent=2))
if __name__=='__main__':main()
