#!/usr/bin/env python3
"""Common-input PINN4SOH-style baseline.

Uses only the shared raw V/I/time curve. The physics term is the documented
capacity-degradation prior: predicted SOH should not increase with cycle order;
capacity-regeneration points remain in the labels and are not deleted.
"""
import csv,json,copy,os
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'/os.environ.get('PINN_OUT','pinn4soh_common_input');K=10;SEED_START=int(os.environ.get('SEED_START','0'));SEED_COUNT=int(os.environ.get('SEED_COUNT','5'));SEEDS=list(range(SEED_START,SEED_START+SEED_COUNT))
class Net(nn.Module):
 def __init__(self): super().__init__();self.f=nn.Sequential(nn.Flatten(),nn.Linear(384,128),nn.GELU(),nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
 def forward(self,x):return self.f(x)
def phy(p): return torch.relu(p[1:]-p[:-1]).mean() if len(p)>1 else p.sum()*0
def main():
 OUT.mkdir(parents=True,exist_ok=True);source,target=formal.read_cells(formal.SOURCE_ZIP),formal.read_cells(formal.TARGET_ZIP);tr=[c for i,c in enumerate(source) if i%5!=4];va=[c for i,c in enumerate(source) if i%5==4];sx,sy=np.concatenate([c[1] for c in tr]),np.concatenate([c[2] for c in tr]);vx=np.concatenate([c[1] for c in va]);mu,sd=sx.mean((0,2),keepdims=True),sx.std((0,2),keepdims=True);sd[sd<1e-6]=1;norm=lambda x:((x-mu)/sd).astype('float32');sx=norm(sx);vx=norm(vx);keep=np.linspace(0,len(sx)-1,min(1000,len(sx))).astype(int);sx,sy=sx[keep],sy[keep];target=[(n,norm(x),y) for n,x,y in target];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[];cells=[]
 for seed in SEEDS:
  print(f'PINN4SOH common-input seed={seed}',flush=True);torch.manual_seed(seed);model=Net().to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3);xb=torch.from_numpy(sx).to(device);yb=torch.from_numpy(sy[:,None]).float().to(device)
  for _ in range(25):
   p=model(xb);loss=nn.functional.mse_loss(p,yb);opt.zero_grad();loss.backward();opt.step()
  maes=[];rmses=[]
  for cid,x,y in target:
   local=copy.deepcopy(model);o=torch.optim.AdamW(local.parameters(),lr=5e-4);xs=torch.from_numpy(x[:K]).to(device);ys=torch.from_numpy(y[:K,None]).float().to(device)
   for _ in range(30):
    ps=local(xs);loss=nn.functional.mse_loss(ps,ys)+0.2*phy(ps);o.zero_grad();loss.backward();o.step()
   with torch.no_grad():p=local(torch.from_numpy(x[K:]).to(device)).cpu().numpy().ravel()
   a=float(mean_absolute_error(y[K:],p));b=float(mean_squared_error(y[K:],p)**.5);maes.append(a);rmses.append(b);cells.append({'track':'specialized_common_input','model':'PINN4SOH_common_input','variant':'support_adaptation','seed':seed,'K':K,'cell_id':cid,'mae':a,'rmse':b})
  rows.append({'track':'specialized_common_input','model':'PINN4SOH_common_input','variant':'support_adaptation','seed':seed,'K':K,'macro_mae':float(np.mean(maes)),'macro_rmse':float(np.mean(rmses)),'target_cells':len(maes),'input':'raw_V_I_time','query_labels_used_for_training':False})
 for fn,data in [('pinn_results.csv',rows),('pinn_cell_results.csv',cells)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
 (OUT/'manifest.json').write_text(json.dumps({'task':'XJTU_to_HUST','K':K,'seeds':SEEDS,'input':'common raw V/I/time only','physics_term':'relu(predicted_SOH_next-predicted_SOH_previous)','reference':'PINN4SOH official repository; independent common-input reimplementation','query_labels_used_for_training':False},indent=2),encoding='utf8')
if __name__=='__main__':main()
