#!/usr/bin/env python3
"""Source-only raw-curve transfer screen: XJTU source -> HUST target.

Source validation controls early stopping.  Target labels are used only for
the final domain-level score, so this is a development transfer diagnostic,
not a tuned result for an external test set.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_raw_curve_screen as base

ROOT=HERE.parent
SOURCE=ROOT/'数据集/BatteryLife_v12_processed/XJTU.zip'
TARGET=ROOT/'数据集/BatteryLife_v12_processed/HUST.zip'
OUT=HERE/'results'

def score(model,loader,y,owners,device):
    model.eval()
    with torch.no_grad(): pred=np.concatenate([model(x.to(device)).cpu().numpy().ravel() for x,_ in loader])
    d={}
    for o,a,b in zip(owners,y,pred): d.setdefault(o,([],[])); d[o][0].append(a); d[o][1].append(b)
    return float(np.mean([mean_absolute_error(a,b) for a,b in d.values()])),float(np.mean([mean_squared_error(a,b)**.5 for a,b in d.values()]))

def main():
    cells_s=base.read_cells(SOURCE); cells_t=base.read_cells(TARGET)
    s_train=[c for i,c in enumerate(cells_s) if i%5!=4]; s_valid=[c for i,c in enumerate(cells_s) if i%5==4]
    tx=np.concatenate([c[1] for c in s_train]); ty=np.concatenate([c[2] for c in s_train])
    sxv=np.concatenate([c[1] for c in s_valid]); syv=np.concatenate([c[2] for c in s_valid]); sov=sum(([c[0]]*len(c[2]) for c in s_valid),[])
    hx=np.concatenate([c[1] for c in cells_t]); hy=np.concatenate([c[2] for c in cells_t]); hov=sum(([c[0]]*len(c[2]) for c in cells_t),[])
    mu=tx.mean((0,2),keepdims=True); sd=tx.std((0,2),keepdims=True); sd[sd<1e-6]=1
    tx=((tx-mu)/sd).astype('float32'); sxv=((sxv-mu)/sd).astype('float32'); hx=((hx-mu)/sd).astype('float32')
    tr=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty[:,None])),batch_size=1024,shuffle=True)
    va=DataLoader(TensorDataset(torch.from_numpy(sxv),torch.from_numpy(syv[:,None])),batch_size=2048)
    ta=DataLoader(TensorDataset(torch.from_numpy(hx),torch.from_numpy(hy[:,None])),batch_size=2048)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]
    for name,ctor in base.MODELS.items():
      for seed in (0,1):
        base.seed(seed); m=ctor().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4); best=1e9; state=None; stale=0
        for ep in range(25):
          m.train()
          for x,y in tr:
            loss=nn.functional.mse_loss(m(x.to(dev)),y.to(dev)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
          vmae,_=score(m,va,syv,sov,dev)
          if vmae<best: best=vmae; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; stale=0
          else: stale+=1
          if stale>=7: break
        m.load_state_dict(state); smae,srmse=score(m,va,syv,sov,dev); tmae,trmse=score(m,ta,hy,hov,dev)
        row={'model':name,'seed':seed,'source_val_macro_mae':smae,'target_hust_macro_mae':tmae,'source_val_macro_rmse':srmse,'target_hust_macro_rmse':trmse,'epochs':ep+1,'parameters':sum(p.numel() for p in m.parameters())}; print(row,flush=True); rows.append(row)
    OUT.mkdir(exist_ok=True)
    (OUT/'cross_domain_raw_split_manifest.json').write_text(json.dumps({'source':str(SOURCE),'target':str(TARGET),'source_train_cells':[c[0] for c in s_train],'source_validation_cells':[c[0] for c in s_valid],'target_all_cells':[c[0] for c in cells_t],'target_labels_used_only_for_score':True},indent=2),encoding='utf8')
    with (OUT/'cross_domain_raw_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,rows[0]); w.writeheader(); w.writerows(rows)
    summary=[]
    for n in base.MODELS:
      q=[r for r in rows if r['model']==n]; summary.append({'model':n,'target_mean_mae':float(np.mean([r['target_hust_macro_mae'] for r in q])),'target_std_mae':float(np.std([r['target_hust_macro_mae'] for r in q])),'target_mean_rmse':float(np.mean([r['target_hust_macro_rmse'] for r in q])),'source_mean_mae':float(np.mean([r['source_val_macro_mae'] for r in q])),'parameters':q[0]['parameters']})
    summary.sort(key=lambda r:(r['target_mean_mae'],r['target_std_mae'])); (OUT/'cross_domain_raw_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
