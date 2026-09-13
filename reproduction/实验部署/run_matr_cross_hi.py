#!/usr/bin/env python3
"""Use the converted MATR cells in a signal-only HI cross-dataset track."""
import csv,json,pickle
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal
from run_remaining_baselines import features
ROOT=Path(__file__).resolve().parents[1];MATR=ROOT/'数据集/MATR_BatteryML_processed';OUT=Path(__file__).resolve().parent/'results'/'matr_cross_hi';K=10;SEEDS=[0,1,2,3,4]
def load_matr():
 out=[]
 for p in sorted(MATR.glob('*.pkl')):
  o=pickle.load(p.open('rb')); nom=float(o.get('nominal_capacity_in_Ah',1.0));xs=[];ys=[]
  for c in sorted(o['cycle_data'],key=lambda z:float(z.get('cycle_number',0))):
   cur=np.asarray(c['current_in_A'],dtype='float32');vol=np.asarray(c['voltage_in_V'],dtype='float32');tim=np.asarray(c['time_in_s'],dtype='float32');cap=np.asarray(c['discharge_capacity_in_Ah'],dtype='float32');pos=np.flatnonzero(cur>0)
   if len(pos)<32 or not len(cap) or float(np.nanmax(cap))<=0:continue
   a,b=int(pos[0]),int(pos[-1])+1;grid=np.linspace(0,1,128,dtype='float32');old=np.linspace(0,1,b-a,dtype='float32');xx=np.stack([np.interp(grid,old,z[a:b]) for z in (vol,cur,tim)]);xx[2]=(xx[2]-xx[2,0])/(xx[2,-1]-xx[2,0]+1e-6);xs.append(xx);ys.append(float(np.nanmax(cap)/nom))
  if xs:out.append((o.get('cell_id',p.stem),np.stack(xs).astype('float32'),np.asarray(ys,dtype='float32')))
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);matr=load_matr();hust=formal.read_cells(ROOT/'数据集/BatteryLife_v12_processed/HUST.zip');rows=[];cells=[]
 for source_name,target_name,source,target in [('MATR','HUST',matr,hust),('HUST','MATR',hust,matr)]:
  print(f'{source_name}->{target_name}',flush=True);src=[c for i,c in enumerate(source) if i%5!=4];tx=np.concatenate([c[1] for c in src]);ty=np.concatenate([c[2] for c in src]);keep=np.linspace(0,len(tx)-1,min(1000,len(tx))).astype(int);sh=features(tx[keep]);ty=ty[keep];th=[(n,features(x),y) for n,x,y in target]
  for seed in SEEDS:
   est=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(sh,ty);ma=[];rm=[]
   for cid,h,y in th:
    if len(y)<=K:continue
    p=est.predict(h[K:]).reshape(-1);a=float(mean_absolute_error(y[K:],p));b=float(mean_squared_error(y[K:],p)**.5);ma.append(a);rm.append(b);cells.append({'source':source_name,'target':target_name,'model':'Ridge','seed':seed,'K':K,'cell_id':cid,'mae':a,'rmse':b})
   rows.append({'source':source_name,'target':target_name,'model':'Ridge','seed':seed,'K':K,'macro_mae':float(np.mean(ma)),'macro_rmse':float(np.mean(rm)),'target_cells':len(ma)})
 for fn,data in [('matr_results.csv',rows),('matr_cell_results.csv',cells)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
 (OUT/'manifest.json').write_text(json.dumps({'source':'BatteryML MATR converted cells','targets':['HUST','MATR'],'K':K,'seeds':SEEDS,'input':'same resampled V/I/time -> locked 16 HI features'},indent=2),encoding='utf8');print(json.dumps({'rows':len(rows),'cell_rows':len(cells)},indent=2))
if __name__=='__main__':main()
