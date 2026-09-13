#!/usr/bin/env python3
"""Strict full-target upper bound: train and test on disjoint target cells."""
import csv,json,os
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'/os.environ.get('UPPER_OUT','full_target_heldout_upper');SEEDS=list(range(10));EPOCHS=25
def main():
 OUT.mkdir(parents=True,exist_ok=True);cells=formal.read_cells(formal.TARGET_ZIP);train=[c for i,c in enumerate(cells) if i%5!=4];test=[c for i,c in enumerate(cells) if i%5==4];tx=np.concatenate([c[1] for c in train]);ty=np.concatenate([c[2] for c in train]);mu,sd=tx.mean((0,2),keepdims=True),tx.std((0,2),keepdims=True);sd[sd<1e-6]=1;norm=lambda x:((x-mu)/sd).astype('float32');tx=norm(tx);test=[(n,norm(x),y) for n,x,y in test];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[];out=[]
 for seed in SEEDS:
  print(f'FULL_TARGET_UPPER seed={seed}',flush=True);model=formal.fit_nn(formal.MLP,tx,ty,tx[:1],ty[:1],device,EPOCHS,seed,early_stop=False);ma=[];rm=[]
  for cid,x,y in test:
   p=formal.predict(model,x,device);a=float(mean_absolute_error(y,p));b=float(mean_squared_error(y,p)**.5);ma.append(a);rm.append(b);out.append({'track':'raw_V_I_time','model':'MLP','variant':'full_target_upper_bound_heldout_cells','seed':seed,'cell_id':cid,'mae':a,'rmse':b})
  rows.append({'track':'raw_V_I_time','model':'MLP','variant':'full_target_upper_bound_heldout_cells','seed':seed,'macro_mae':float(np.mean(ma)),'macro_rmse':float(np.mean(rm)),'train_target_cells':len(train),'heldout_target_cells':len(test),'query_labels_used_for_training':False})
 for fn,data in [('upper_results.csv',rows),('upper_cell_results.csv',out)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
 (OUT/'manifest.json').write_text(json.dumps({'task':'HUST_within_target_cell_holdout','train_cells':[c[0] for c in train],'heldout_cells':[c[0] for c in test],'split_unit':'cell','note':'upper bound only; not comparable to XJTU-to-HUST transfer'},indent=2),encoding='utf8')
if __name__=='__main__':main()
