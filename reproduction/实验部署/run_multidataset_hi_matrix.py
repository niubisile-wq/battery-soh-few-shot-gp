#!/usr/bin/env python3
"""Executable multi-dataset allocation matrix before new-model work.

This is deliberately a transparent HI-track matrix: it exercises the same
K=10 chronological protocol on bidirectional core pairs and external targets,
without mixing it with the XJTU->HUST raw-model leaderboard.
"""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal
from run_remaining_baselines import features
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'results'/'multidataset_hi_matrix';K=10;SEEDS=[0,1,2,3,4]
Z={n:ROOT/f'数据集/BatteryLife_v12_processed/{n}.zip' for n in ['HUST','XJTU','Tongji','SNL','CALCE']}
PAIRS=[('XJTU','HUST'),('HUST','XJTU'),('XJTU','Tongji'),('Tongji','XJTU'),('HUST','Tongji'),('Tongji','HUST'),('XJTU','SNL'),('XJTU','CALCE'),('HUST','SNL'),('HUST','CALCE')]
def metric(y,p):return float(mean_absolute_error(y,p)),float(mean_squared_error(y,p)**.5)
def main():
 OUT.mkdir(parents=True,exist_ok=True);cache={};rows=[];cells=[]
 for source_name,target_name in PAIRS:
  print(f'MATRIX {source_name}->{target_name}',flush=True);source=cache.setdefault(source_name,formal.read_cells(Z[source_name]));target=cache.setdefault(target_name,formal.read_cells(Z[target_name]));src=[c for i,c in enumerate(source) if i%5!=4];sx=np.concatenate([c[1] for c in src]);sy=np.concatenate([c[2] for c in src]);mu,sd=sx.mean((0,2),keepdims=True),sx.std((0,2),keepdims=True);sd[sd<1e-6]=1;norm=lambda x:((x-mu)/sd).astype('float32');sh=features(norm(sx));target_h=[(n,features(norm(x)),y) for n,x,y in target]
  for model_name in ['Ridge','last_support','linear_trend']:
   for seed in SEEDS:
    ma=[];rm=[]
    for cid,h,y in target_h:
     qy=y[K:]
     if model_name=='Ridge':
      est=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(sh,sy);p=est.predict(h[K:]).reshape(-1)
     elif model_name=='last_support':p=np.full(len(qy),float(y[K-1]))
     else:p=formal.linear_trend(None,y[:K],len(qy),float(sy.mean()))
     a,b=metric(qy,p);ma.append(a);rm.append(b);cells.append({'source':source_name,'target':target_name,'track':'handcrafted_HI','model':model_name,'variant':'source_only_or_support','seed':seed,'K':K,'cell_id':cid,'mae':a,'rmse':b})
    rows.append({'source':source_name,'target':target_name,'track':'handcrafted_HI','model':model_name,'variant':'source_only_or_support','seed':seed,'K':K,'macro_mae':float(np.mean(ma)),'macro_rmse':float(np.mean(rm)),'target_cells':len(ma)})
 for fn,data in [('matrix_results.csv',rows),('matrix_cell_results.csv',cells)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
 (OUT/'manifest.json').write_text(json.dumps({'pairs':PAIRS,'K':K,'seeds':SEEDS,'protocol':'split source cells before windows; first K target cycles support; later cycles query','datasets_used':sorted(set(sum(([a,b] for a,b in PAIRS),[])))},indent=2),encoding='utf8');print(json.dumps({'rows':len(rows),'cell_rows':len(cells)},indent=2))
if __name__=='__main__':main()
