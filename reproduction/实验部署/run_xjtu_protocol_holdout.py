#!/usr/bin/env python3
"""XJTU leave-one-protocol-out HI control."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal
from run_remaining_baselines import features
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'results'/'xjtu_protocol_holdout';K=10;SEEDS=[0,1,2,3,4]
def group(n):
 for g in ['2C','3C','R2.5','R3','RW']:
  if g in n:return g
 return 'other'
def main():
 OUT.mkdir(parents=True,exist_ok=True);allc=formal.read_cells(formal.SOURCE_ZIP);groups={group(n):[] for n,_,_ in allc};
 for c in allc:groups[group(c[0])].append(c)
 rows=[];cells=[]
 for target_group,target_cells in groups.items():
  source=[c for g,v in groups.items() if g!=target_group for c in v]
  if not source or not target_cells:continue
  print(f'PROTOCOL {target_group}',flush=True);sx=np.concatenate([c[1] for c in source]);sy=np.concatenate([c[2] for c in source]);keep=np.linspace(0,len(sx)-1,min(1000,len(sx))).astype(int);sh=features(sx[keep]);sy=sy[keep];th=[(n,features(x),y) for n,x,y in target_cells]
  for seed in SEEDS:
   est=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(sh,sy);ma=[];rm=[]
   for cid,h,y in th:
    if len(y)<=K:continue
    p=est.predict(h[K:]).reshape(-1);a=float(mean_absolute_error(y[K:],p));b=float(mean_squared_error(y[K:],p)**.5);ma.append(a);rm.append(b);cells.append({'heldout_protocol':target_group,'model':'Ridge','seed':seed,'K':K,'cell_id':cid,'mae':a,'rmse':b})
   rows.append({'heldout_protocol':target_group,'model':'Ridge','seed':seed,'K':K,'macro_mae':float(np.mean(ma)),'macro_rmse':float(np.mean(rm)),'target_cells':len(ma)})
 for fn,d in [('protocol_results.csv',rows),('protocol_cell_results.csv',cells)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=d[0].keys());w.writeheader();w.writerows(d)
 (OUT/'manifest.json').write_text(json.dumps({'protocol_groups':sorted(groups),'split':'leave one protocol group out','K':K,'seeds':SEEDS},indent=2),encoding='utf8');print(json.dumps({'rows':len(rows),'cell_rows':len(cells)},indent=2))
if __name__=='__main__':main()
