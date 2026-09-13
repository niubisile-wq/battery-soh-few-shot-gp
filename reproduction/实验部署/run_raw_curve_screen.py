#!/usr/bin/env python3
"""Small raw charge-curve backbone screen on the official XJTU archive."""
from __future__ import annotations
import argparse, csv, json, random, zipfile, pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parents[1]
ZIP=ROOT/'数据集/BatteryLife_v12_processed/XJTU.zip'
OUT=Path(__file__).resolve().parent/'results'

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def read_cells(zip_path=ZIP):
    cells=[]
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(x for x in z.namelist() if x.endswith('.pkl')):
            with z.open(name) as f: obj=pickle.load(f)
            samples=[]
            for c in obj['cycle_data']:
                cur=np.asarray(c['current_in_A'],dtype=np.float32)
                vol=np.asarray(c['voltage_in_V'],dtype=np.float32)
                tim=np.asarray(c['time_in_s'],dtype=np.float32)
                cap=np.asarray(c['discharge_capacity_in_Ah'],dtype=np.float32)
                # Charge-only input: retain the initial positive-current segment.
                pos=np.flatnonzero(cur>0)
                if len(pos)<32 or len(cap)==0 or float(cap.max())<=0: continue
                end=int(pos[-1])+1; start=int(pos[0]);
                if end-start<32: continue
                grid=np.linspace(0,1,128,dtype=np.float32)
                old=np.linspace(0,1,end-start,dtype=np.float32)
                xx=np.stack([np.interp(grid,old,a[start:end]) for a in (vol,cur,tim)])
                xx[2]=(xx[2]-xx[2,0])/(xx[2,-1]-xx[2,0]+1e-6)
                samples.append((xx,float(cap.max()/obj['nominal_capacity_in_Ah'])))
            if samples: cells.append((name,np.stack([x for x,_ in samples]),np.array([y for _,y in samples],np.float32)))
    return cells

class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(3*128,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x)
class Conv(nn.Module):
    def __init__(self, dilated=False):
        super().__init__(); ds=[1,2,4] if dilated else [1,1,1]
        self.net=nn.Sequential(nn.Conv1d(3,32,5,padding=2*ds[0],dilation=ds[0]),nn.GELU(),nn.Conv1d(32,64,5,padding=2*ds[1],dilation=ds[1]),nn.GELU(),nn.Conv1d(64,64,5,padding=2*ds[2],dilation=ds[2]),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(64,1))
    def forward(self,x): return self.net(x)
class RNN(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn=nn.LSTM(3,64,2,batch_first=True,dropout=.1); self.head=nn.Linear(64,1)
    def forward(self,x): return self.head(self.rnn(x.transpose(1,2))[0][:,-1])
class Attn(nn.Module):
    def __init__(self):
        super().__init__(); self.e=nn.Linear(3,32); layer=nn.TransformerEncoderLayer(32,4,64,.1,batch_first=True,norm_first=True); self.t=nn.TransformerEncoder(layer,2); self.h=nn.Linear(32,1)
    def forward(self,x): return self.h(self.t(self.e(x.transpose(1,2))).mean(1))
class PatchTSTLite(nn.Module):
    """Small patch-token Transformer with the same 128-point input budget."""
    def __init__(self):
        super().__init__(); self.patch=16; self.stride=8
        self.e=nn.Linear(3*self.patch,32)
        layer=nn.TransformerEncoderLayer(32,4,64,.1,batch_first=True,norm_first=True)
        self.t=nn.TransformerEncoder(layer,2); self.h=nn.Linear(32,1)
    def forward(self,x):
        z=x.unfold(-1,self.patch,self.stride).transpose(1,2).flatten(2)
        return self.h(self.t(self.e(z)).mean(1))
MODELS={'MLP':MLP,'CNN1D':lambda:Conv(False),'TCN':lambda:Conv(True),'LSTM':RNN,'Transformer':Attn,'PatchTSTLite':PatchTSTLite}

def score(model,loader,y,owners,device):
    model.eval()
    with torch.no_grad(): pred=np.concatenate([model(x.to(device)).cpu().numpy().ravel() for x,_ in loader])
    d={}
    for o,a,b in zip(owners,y,pred): d.setdefault(o,([],[])); d[o][0].append(a); d[o][1].append(b)
    return float(np.mean([mean_absolute_error(a,b) for a,b in d.values()])),float(np.mean([mean_squared_error(a,b)**.5 for a,b in d.values()]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--seeds',default='0,1,2'); args=ap.parse_args()
    cells=read_cells(); train=[c for i,c in enumerate(cells) if i%5!=4]; valid=[c for i,c in enumerate(cells) if i%5==4]
    tx=np.concatenate([c[1] for c in train]); ty=np.concatenate([c[2] for c in train]); vx=np.concatenate([c[1] for c in valid]); vy=np.concatenate([c[2] for c in valid]); owners=sum(([c[0]]*len(c[2]) for c in valid),[])
    mu=tx.mean((0,2),keepdims=True); sd=tx.std((0,2),keepdims=True); sd[sd<1e-6]=1; tx=((tx-mu)/sd).astype('float32'); vx=((vx-mu)/sd).astype('float32')
    OUT.mkdir(exist_ok=True); manifest={'archive':str(ZIP),'input':'charge-only voltage,current,normalized-time resampled to 128','train_cells':[c[0] for c in train],'validation_cells':[c[0] for c in valid],'test_used':False,'seeds':[int(s) for s in args.seeds.split(',')]}; (OUT/'raw_split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    tr=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty[:,None])),batch_size=512,shuffle=True); va=DataLoader(TensorDataset(torch.from_numpy(vx),torch.from_numpy(vy[:,None])),batch_size=1024)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]
    for name,ctor in MODELS.items():
      for s in manifest['seeds']:
        seed(s); m=ctor().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4); best=1e9; state=None; stale=0
        for ep in range(args.epochs):
          m.train()
          for x,y in tr:
            loss=nn.functional.mse_loss(m(x.to(dev)),y.to(dev)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
          mae,_=score(m,va,vy,owners,dev)
          if mae<best: best=mae; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; stale=0
          else: stale+=1
          if stale>=8: break
        m.load_state_dict(state); mae,rmse=score(m,va,vy,owners,dev); row={'model':name,'seed':s,'val_macro_mae':mae,'val_macro_rmse':rmse,'epochs':ep+1,'parameters':sum(p.numel() for p in m.parameters())}; print(row,flush=True); rows.append(row)
    with (OUT/'raw_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,rows[0]); w.writeheader(); w.writerows(rows)
    summary=[]
    for n in MODELS:
      q=[r for r in rows if r['model']==n]; summary.append({'model':n,'mean_mae':float(np.mean([r['val_macro_mae'] for r in q])),'std_mae':float(np.std([r['val_macro_mae'] for r in q])),'mean_rmse':float(np.mean([r['val_macro_rmse'] for r in q])),'parameters':q[0]['parameters']})
    summary.sort(key=lambda r:(r['mean_mae'],r['std_mae'])); (OUT/'raw_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
