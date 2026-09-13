#!/usr/bin/env python3
"""Unsupervised domain-adaptation baselines on the locked raw-input task.

Target SOH labels are never used. Target signals are used only as unlabeled
domain observations; evaluation remains on the later K=10 query cycles.
"""
import csv, json, copy, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import run_formal_protocol as formal

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"/os.environ.get("UDA_OUT","uda_separate")
K=10; SEED_START=int(os.environ.get("SEED_START","0")); SEED_COUNT=int(os.environ.get("SEED_COUNT","5")); SEEDS=list(range(SEED_START,SEED_START+SEED_COUNT)); EPOCHS=20

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.feat=nn.Sequential(nn.Flatten(),nn.Linear(384,128),nn.GELU(),nn.Linear(128,32),nn.GELU()); self.head=nn.Linear(32,1)
    def forward(self,x): return self.head(self.feat(x))

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam=lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -ctx.lam*grad, None

def coral(a,b):
    a=a-a.mean(0,keepdim=True); b=b-b.mean(0,keepdim=True)
    ca=(a.T@a)/max(1,len(a)-1); cb=(b.T@b)/max(1,len(b)-1)
    return ((ca-cb)**2).mean()
def mmd(a,b):
    # Linear-kernel MMD is deterministic and scales safely for this screening.
    return ((a.mean(0)-b.mean(0))**2).mean()
def run(method, seed, sx, sy, tx, device):
    torch.manual_seed(seed); np.random.seed(seed); model=Net().to(device)
    xs=torch.from_numpy(sx).to(device); ys=torch.from_numpy(sy[:,None]).to(device); xt=torch.from_numpy(tx).to(device)
    n=min(len(xs),len(xt)); g=torch.Generator(device="cpu").manual_seed(seed)
    domain = nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,2)).to(device) if method=="DANN" else None
    params=list(model.parameters())+(list(domain.parameters()) if domain is not None else [])
    opt=torch.optim.AdamW(params,lr=2e-3,weight_decay=1e-4)
    for _ in range(EPOCHS):
        si=torch.randperm(len(xs),generator=g)[:n]; ti=torch.randperm(len(xt),generator=g)[:n]
        fs=model.feat(xs[si]); ft=model.feat(xt[ti]); loss=nn.functional.mse_loss(model.head(fs),ys[si])
        if method=="MMD": loss=loss+0.1*mmd(fs,ft)
        elif method=="DeepCORAL": loss=loss+0.1*coral(fs,ft)
        elif method=="DANN":
            dom_x=torch.cat([fs,ft],0); dom_y=torch.cat([torch.zeros(len(fs),dtype=torch.long,device=device),torch.ones(len(ft),dtype=torch.long,device=device)])
            loss=loss+0.1*nn.functional.cross_entropy(domain(GRL.apply(dom_x,1.0)),dom_y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model
def main():
    OUT.mkdir(parents=True,exist_ok=True); source,target=formal.read_cells(formal.SOURCE_ZIP),formal.read_cells(formal.TARGET_ZIP)
    tr=[c for i,c in enumerate(source) if i%5!=4]; va=[c for i,c in enumerate(source) if i%5==4]
    sx,sy=np.concatenate([c[1] for c in tr]),np.concatenate([c[2] for c in tr]); vx=np.concatenate([c[1] for c in va])
    mu,sd=sx.mean((0,2),keepdims=True),sx.std((0,2),keepdims=True); sd[sd<1e-6]=1; norm=lambda x:((x-mu)/sd).astype('float32')
    sx=norm(sx); vx=norm(vx); keep=np.linspace(0,len(sx)-1,min(1000,len(sx))).astype(int); sx,sy=sx[keep],sy[keep]
    target=[(n,norm(x),y) for n,x,y in target]; unlabeled=np.concatenate([x for _,x,_ in target]); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]; cells=[]
    for method in ('MMD','DeepCORAL','DANN'):
      for seed in SEEDS:
        print(f'UDA {method} seed={seed}',flush=True); model=run(method,seed,sx,sy,unlabeled,device); maes=[];rmses=[]
        for cid,x,y in target:
          with torch.no_grad(): p=model(torch.from_numpy(x[K:]).to(device)).cpu().numpy().ravel()
          a=float(mean_absolute_error(y[K:],p)); b=float(mean_squared_error(y[K:],p)**.5); maes.append(a);rmses.append(b);cells.append({'track':'uda_separate','model':method,'variant':'unlabeled_target_adaptation','seed':seed,'K':K,'cell_id':cid,'mae':a,'rmse':b})
        rows.append({'track':'uda_separate','model':method,'variant':'unlabeled_target_adaptation','seed':seed,'K':K,'macro_mae':float(np.mean(maes)),'macro_rmse':float(np.mean(rmses)),'target_cells':len(maes),'query_labels_used_for_training':False})
    for fn,data in [('uda_results.csv',rows),('uda_cell_results.csv',cells)]:
      with (OUT/fn).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    (OUT/'manifest.json').write_text(json.dumps({'task':'XJTU_to_HUST','K':K,'methods':['MMD','DeepCORAL','DANN'],'seeds':SEEDS,'target_signals_used_as_unlabeled':True,'target_soh_labels_used_for_training':False,'DANN':'gradient-reversal domain classifier'},indent=2),encoding='utf8')
if __name__=='__main__': main()
