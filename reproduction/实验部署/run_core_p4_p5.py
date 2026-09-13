#!/usr/bin/env python3
"""P4 multi-source and P5 leave-one-dataset-out HI/raw-feature controls."""
import csv,json,pickle
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal
from run_remaining_baselines import features
from run_matr_cross_hi import load_matr
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'results'/'core_p4_p5';K=10;SEEDS=[0,1,2,3,4]
def feat(cells):return [(n,features(x),y) for n,x,y in cells]
def evaluate(source_name,target_name,source,target,track,rows,cells):
 sh=np.concatenate([h for _,h,_ in source]);sy=np.concatenate([y for _,_,y in source]);keep=np.linspace(0,len(sy)-1,min(1000,len(sy))).astype(int);sh,sy=sh[keep],sy[keep];th=target
 for seed in SEEDS:
  est=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(sh,sy);ma=[];rm=[]
  for cid,h,y in th:
   if len(y)<=K:continue
   p=est.predict(h[K:]).reshape(-1);a=float(mean_absolute_error(y[K:],p));b=float(mean_squared_error(y[K:],p)**.5);ma.append(a);rm.append(b);cells.append({'source':source_name,'target':target_name,'track':track,'model':'Ridge','seed':seed,'K':K,'cell_id':cid,'mae':a,'rmse':b})
  rows.append({'source':source_name,'target':target_name,'track':track,'model':'Ridge','seed':seed,'K':K,'macro_mae':float(np.mean(ma)),'macro_rmse':float(np.mean(rm)),'target_cells':len(ma)})
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 z={n:ROOT/f'数据集/BatteryLife_v12_processed/{n}.zip' for n in ['HUST','XJTU','Tongji']};data={'MATR':load_matr()}
 for n,p in z.items():print('READ',n,flush=True);data[n]=formal.read_cells(p)
 data={n:feat(v) for n,v in data.items()};rows=[];cells=[]
 # P4: two-source pooled training to the two remaining core domains.
 evaluate('MATR+HUST','XJTU',data['MATR']+data['HUST'],data['XJTU'],'P4_multisource',rows,cells)
 evaluate('MATR+HUST','Tongji',data['MATR']+data['HUST'],data['Tongji'],'P4_multisource',rows,cells)
 # P5: each core dataset is held out while the other three are pooled.
 core=list(data)
 for target in core:evaluate('+'.join(n for n in core if n!=target),target,[x for n in core if n!=target for x in data[n]],data[target],'P5_leave_one_dataset_out',rows,cells)
 for fn,d in [('p4_p5_results.csv',rows),('p4_p5_cell_results.csv',cells)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=d[0].keys());w.writeheader();w.writerows(d)
 (OUT/'manifest.json').write_text(json.dumps({'P4':['MATR+HUST->XJTU','MATR+HUST->Tongji'],'P5':'each of MATR/HUST/XJTU/Tongji held out; other three pooled','K':K,'seeds':SEEDS,'track':'common signal-derived HI'},indent=2),encoding='utf8');print(json.dumps({'rows':len(rows),'cell_rows':len(cells)},indent=2))
if __name__=='__main__':main()
