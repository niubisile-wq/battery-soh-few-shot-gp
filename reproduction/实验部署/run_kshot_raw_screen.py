#!/usr/bin/env python3
"""K=10 target-support raw-curve screen: XJTU pretrain -> HUST adaptation."""
from __future__ import annotations
import argparse, copy, csv, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); import run_raw_curve_screen as base
ROOT=HERE.parent; SOURCE=ROOT/'数据集/BatteryLife_v12_processed/XJTU.zip'; TARGET=ROOT/'数据集/BatteryLife_v12_processed/HUST.zip'; OUT=HERE/'results'

def cell_score(model,x,y,device):
    model.eval()
    with torch.no_grad(): p=model(torch.from_numpy(x).to(device)).cpu().numpy().ravel()
    return mean_absolute_error(y,p),mean_squared_error(y,p)**.5

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default=str(SOURCE)); ap.add_argument('--target',default=str(TARGET)); ap.add_argument('--prefix',default='k10_raw')
    args=ap.parse_args(); K=10
    source_zip=Path(args.source); target_zip=Path(args.target)
    source=base.read_cells(source_zip); target=base.read_cells(target_zip)
    src_tr=[c for i,c in enumerate(source) if i%5!=4]; src_va=[c for i,c in enumerate(source) if i%5==4]
    tx=np.concatenate([c[1] for c in src_tr]); ty=np.concatenate([c[2] for c in src_tr]); vx=np.concatenate([c[1] for c in src_va]); vy=np.concatenate([c[2] for c in src_va])
    mu=tx.mean((0,2),keepdims=True); sd=tx.std((0,2),keepdims=True); sd[sd<1e-6]=1
    tx=((tx-mu)/sd).astype('float32'); vx=((vx-mu)/sd).astype('float32')
    target_arrays=[]
    for name,x,y in target:
        z=((x-mu)/sd).astype('float32')
        if len(y)>K: target_arrays.append((name,z,y))
    tr=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty[:,None])),batch_size=1024,shuffle=True)
    va=DataLoader(TensorDataset(torch.from_numpy(vx),torch.from_numpy(vy[:,None])),batch_size=2048)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]
    for name,ctor in base.MODELS.items():
      for seed in (0,1):
        base.seed(seed); model=ctor().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); best=1e9; state=None; stale=0
        for ep in range(25):
          model.train()
          for x,y in tr:
            loss=nn.functional.mse_loss(model(x.to(dev)),y.to(dev)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
          model.eval()
          with torch.no_grad(): pred=np.concatenate([model(x.to(dev)).cpu().numpy().ravel() for x,_ in va])
          # Source validation is window-level only for early stopping; target is untouched.
          v=float(np.mean((pred-vy)**2))
          if v<best: best=v; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
          else: stale+=1
          if stale>=7: break
        model.load_state_dict(state); maes=[]; rmses=[]
        for cell_name,x,y in target_arrays:
          adapted=copy.deepcopy(model).to(dev); aopt=torch.optim.AdamW(adapted.parameters(),lr=5e-4,weight_decay=1e-5)
          xs=torch.from_numpy(x[:K]); ys=torch.from_numpy(y[:K,None]);
          # Fixed adaptation budget; no query labels or query early stopping.
          adapted.train()
          for _ in range(30):
            loss=nn.functional.mse_loss(adapted(xs.to(dev)),ys.to(dev)); aopt.zero_grad(set_to_none=True); loss.backward(); aopt.step()
          mae,rmse=cell_score(adapted,x[K:],y[K:],dev); maes.append(mae); rmses.append(rmse)
        row={'model':name,'seed':seed,'K':K,'target_hust_macro_mae':float(np.mean(maes)),'target_hust_macro_rmse':float(np.mean(rmses)),'target_cells':len(maes),'source_train_epochs':ep+1,'adaptation_steps':30,'parameters':sum(p.numel() for p in model.parameters())}; print(row,flush=True); rows.append(row)
    OUT.mkdir(exist_ok=True); (OUT/f'{args.prefix}_split_manifest.json').write_text(json.dumps({'source':str(source_zip),'target':str(target_zip),'K':K,'source_train_cells':[c[0] for c in src_tr],'source_validation_cells':[c[0] for c in src_va],'target_cells':[c[0] for c in target_arrays],'support':'first K chronological cycles','query_labels_used_only_for_score':True,'adaptation_steps':30},indent=2),encoding='utf8')
    with (OUT/f'{args.prefix}_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,rows[0]); w.writeheader(); w.writerows(rows)
    summary=[]
    for n in base.MODELS:
      q=[r for r in rows if r['model']==n]; summary.append({'model':n,'target_mean_mae':float(np.mean([r['target_hust_macro_mae'] for r in q])),'target_std_mae':float(np.std([r['target_hust_macro_mae'] for r in q])),'target_mean_rmse':float(np.mean([r['target_hust_macro_rmse'] for r in q])),'parameters':q[0]['parameters']})
    summary.sort(key=lambda r:(r['target_mean_mae'],r['target_std_mae'])); (OUT/f'{args.prefix}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
