#!/usr/bin/env python3
"""Within-dataset cell-holdout HI baselines (P1-style control)."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal
from run_remaining_baselines import features
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'results'/'self_dataset_hi';K=10;SEEDS=[0,1,2,3,4];DATA=['XJTU','HUST','Tongji']
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];cells=[]
 for name in DATA:
  print(f'SELF {name}',flush=True);allc=formal.read_cells(ROOT/f'数据集/BatteryLife_v12_processed/{name}.zip');src=[c for i,c in enumerate(allc) if i%5!=4];test=[c for i,c in enumerate(allc) if i%5==4];sx=np.concatenate([c[1] for c in src]);sy=np.concatenate([c[2] for c in src]);mu,sd=sx.mean((0,2),keepdims=True),sx.std((0,2),keepdims=True);sd[sd<1e-6]=1;norm=lambda x:((x-mu)/sd).astype('float32');keep=np.linspace(0,len(sx)-1,min(1000,len(sx))).astype(int);sh=features(norm(sx[keep]));sy=sy[keep];th=[(n,features(norm(x)),y) for n,x,y in test]
  for model in ['Ridge','last_support','linear_trend']:
   for seed in SEEDS:
    ma=[];rm=[]
    for cid,h,y in th:
     if model=='Ridge':p=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(sh,sy).predict(h[K:]).reshape(-1)
     elif model=='last_support':p=np.full(len(y[K:]),float(y[K-1]))
     else:p=formal.linear_trend(None,y[:K],len(y[K:]),float(sy.mean()))
     a=float(mean_absolute_error(y[K:],p));b=float(mean_squared_error(y[K:],p)**.5);ma.append(a);rm.append(b);cells.append({'dataset':name,'track':'handcrafted_HI','model':model,'seed':seed,'K':K,'cell_id':cid,'mae':a,'rmse':b})
    rows.append({'dataset':name,'track':'handcrafted_HI','model':model,'seed':seed,'K':K,'macro_mae':float(np.mean(ma)),'macro_rmse':float(np.mean(rm)),'source_cells':len(src),'heldout_cells':len(test)})
 for fn,data in [('self_results.csv',rows),('self_cell_results.csv',cells)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
 (OUT/'manifest.json').write_text(json.dumps({'datasets':DATA,'split':'cell-level 80/20, split before windows','K':K,'seeds':SEEDS},indent=2),encoding='utf8');print(json.dumps({'rows':len(rows),'cell_rows':len(cells)},indent=2))
if __name__=='__main__':main()
