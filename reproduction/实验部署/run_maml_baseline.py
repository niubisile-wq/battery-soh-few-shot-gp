#!/usr/bin/env python3
"""Strict K=10 MAML baseline using the same raw V/I/time representation."""
import csv,json,copy,os
from pathlib import Path
import numpy as np, torch
from torch import nn
from sklearn.metrics import mean_absolute_error,mean_squared_error
import run_formal_protocol as formal

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'/os.environ.get('MAML_OUT','maml'); K=10; SEED_START=int(os.environ.get('SEED_START','0')); SEED_COUNT=int(os.environ.get('SEED_COUNT','5')); SEEDS=list(range(SEED_START,SEED_START+SEED_COUNT))
class Net(nn.Module):
 def __init__(self): super().__init__(); self.f=nn.Sequential(nn.Flatten(),nn.Linear(384,128),nn.GELU(),nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
 def forward(self,x): return self.f(x)
def main():
 OUT.mkdir(parents=True,exist_ok=True); source,target=formal.read_cells(formal.SOURCE_ZIP),formal.read_cells(formal.TARGET_ZIP); tr=[c for i,c in enumerate(source) if i%5!=4]; va=[c for i,c in enumerate(source) if i%5==4]
 sx,sy=np.concatenate([c[1] for c in tr]),np.concatenate([c[2] for c in tr]); vx=np.concatenate([c[1] for c in va]); mu,sd=sx.mean((0,2),keepdims=True),sx.std((0,2),keepdims=True);sd[sd<1e-6]=1; norm=lambda x:((x-mu)/sd).astype('float32'); sx=norm(sx);vx=norm(vx);keep=np.linspace(0,len(sx)-1,min(1000,len(sx))).astype(int);sx,sy=sx[keep],sy[keep];target=[(n,norm(x),y) for n,x,y in target];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[];cells=[]
 for seed in SEEDS:
  print(f'MAML seed={seed}',flush=True);torch.manual_seed(seed);np.random.seed(seed);model=Net().to(device);meta=torch.optim.Adam(model.parameters(),lr=1e-3); rng=np.random.default_rng(seed)
  # Each source cell is an episode; labels after the episode support are used
  # only for meta-training, never from the held-out HUST target cells.
  for it in range(120):
   cid,x,y=tr[int(rng.integers(len(tr)))]; x=norm(x); m=min(10,len(y)//2); q=min(20,len(y)-m)
   if q<1: continue
   xb=torch.from_numpy(x[:m]).to(device);yb=torch.from_numpy(y[:m,None]).float().to(device);xq=torch.from_numpy(x[m:m+q]).to(device);yq=torch.from_numpy(y[m:m+q,None]).float().to(device)
   params=dict(model.named_parameters()); fast={k:v for k,v in params.items()}; loss=nn.functional.mse_loss(model(xb),yb); grads=torch.autograd.grad(loss,tuple(fast.values()),create_graph=True)
   fast={k:v-0.01*g for (k,v),g in zip(fast.items(),grads)}; pred=torch.func.functional_call(model,fast,(xq,)); outer=nn.functional.mse_loss(pred,yq);meta.zero_grad();outer.backward();meta.step()
  maes=[];rmses=[]
  for cid,x,y in target:
   local=copy.deepcopy(model);opt=torch.optim.Adam(local.parameters(),lr=5e-4);xb=torch.from_numpy(x[:K]).to(device);yb=torch.from_numpy(y[:K,None]).float().to(device)
   for _ in range(30): opt.zero_grad();loss=nn.functional.mse_loss(local(xb),yb);loss.backward();opt.step()
   with torch.no_grad():p=local(torch.from_numpy(x[K:]).to(device)).cpu().numpy().ravel()
   a=float(mean_absolute_error(y[K:],p));b=float(mean_squared_error(y[K:],p)**.5);maes.append(a);rmses.append(b);cells.append({'track':'raw_V_I_time','model':'MAML','variant':'meta_train_then_K10_adapt','seed':seed,'K':K,'cell_id':cid,'mae':a,'rmse':b})
  rows.append({'track':'raw_V_I_time','model':'MAML','variant':'meta_train_then_K10_adapt','seed':seed,'K':K,'macro_mae':float(np.mean(maes)),'macro_rmse':float(np.mean(rmses)),'target_cells':len(maes),'query_labels_used_for_training':False})
 for fn,data in [('maml_results.csv',rows),('maml_cell_results.csv',cells)]:
  with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
 (OUT/'manifest.json').write_text(json.dumps({'task':'XJTU_to_HUST','K':K,'seeds':SEEDS,'meta_iterations':120,'inner_steps':1,'target_adaptation_steps':30,'query_labels_used_for_target_adaptation':False},indent=2),encoding='utf8')
if __name__=='__main__':main()
