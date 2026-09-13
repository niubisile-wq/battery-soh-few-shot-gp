"""Corrected MAML and separate UDA baselines on the locked caches."""
import argparse, csv, hashlib, json, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]

class Net(nn.Module):
    def __init__(self, domain=False):
        super().__init__(); self.feat=nn.Sequential(nn.Flatten(),nn.Linear(384,128),nn.GELU(),nn.Linear(128,32),nn.GELU()); self.head=nn.Linear(32,1)
    def forward(self,x): return self.head(self.feat(x))

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lam): ctx.lam=lam; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.lam*g,None

def load(base, dirs):
    out=[]
    for d in dirs:
        for p in sorted((base/d).glob('*.npz')):
            with np.load(p,allow_pickle=True) as z:
                x=z['x'].astype('float32');q=z['capacity_Ah'].astype('float32');cyc=z['cycle_number'].astype('float32'); cid=str(z['cell_id'].item())
            o=np.argsort(cyc); out.append({'id':cid,'x':x[o],'y':q[o]/q[o][0],'cycle':cyc[o]})
    return out

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9'); ap.add_argument('--K',type=int,default=10); ap.add_argument('--output',default='corrected_transfer_baselines_v1'); args=ap.parse_args()
    out=ROOT/'开发基线选择依据/results'/args.output
    if out.exists() and any(out.iterdir()): raise RuntimeError(f'refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    src=ROOT/'实验部署/source_window_cache_v2/offset_03_01'; tgt=ROOT/'实验部署/hust_target_cache_v1'; tr=load(src,['source_train']); te=load(tgt,['.'])
    sx=np.concatenate([r['x'] for r in tr]); sy=np.concatenate([r['y'] for r in tr]); keep=np.linspace(0,len(sy)-1,min(1000,len(sy))).astype(int); sx,sy=sx[keep],sy[keep]
    mu=sx.mean((0,2),keepdims=True); sd=sx.std((0,2),keepdims=True); sd[sd<1e-6]=1; norm=lambda a:((a-mu)/sd).astype('float32')
    sx=norm(sx); unlabeled=norm(np.concatenate([r['x'] for r in te]))
    for r in te:r['x']=norm(r['x'])
    seeds=[int(s) for s in args.seeds.split(',')]; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[];cells=[]
    # MAML: source-cell episodes, then exactly K target support adaptation.
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); model=Net().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); rng=np.random.default_rng(seed)
        for _ in range(120):
            r=tr[int(rng.integers(len(tr)))]; x=norm(r['x']); y=r['y']; m=min(10,len(y)//2); q=min(20,len(y)-m)
            if q<1: continue
            xb=torch.from_numpy(x[:m]).to(device); yb=torch.from_numpy(y[:m,None]).float().to(device); xq=torch.from_numpy(x[m:m+q]).to(device); yq=torch.from_numpy(y[m:m+q,None]).float().to(device)
            params=dict(model.named_parameters()); loss=nn.functional.mse_loss(model(xb),yb); grads=torch.autograd.grad(loss,tuple(params.values()),create_graph=True); fast={k:v-.01*g for (k,v),g in zip(params.items(),grads)}
            pred=torch.func.functional_call(model,fast,(xq,)); outer=nn.functional.mse_loss(pred,yq); opt.zero_grad(); outer.backward(); opt.step()
        maes=[];rmses=[]
        for r in te:
            local=copy.deepcopy(model); o=torch.optim.Adam(local.parameters(),lr=5e-4); xb=torch.from_numpy(r['x'][:args.K]).to(device);yb=torch.from_numpy(r['y'][:args.K,None]).float().to(device)
            for _ in range(50): o.zero_grad();loss=nn.functional.mse_loss(local(xb),yb);loss.backward();o.step()
            with torch.no_grad():p=local(torch.from_numpy(r['x'][args.K:]).to(device)).cpu().numpy().ravel()
            y=r['y'][args.K:];mae=float(np.mean(abs(p-y)));rmse=float(np.sqrt(np.mean((p-y)**2)));maes.append(mae);rmses.append(rmse);cells.append({'model':'MAML','seed':seed,'K':args.K,'cell_id':r['id'],'mae':mae,'rmse':rmse})
        rows.append({'model':'MAML','seed':seed,'K':args.K,'macro_mae':float(np.mean(maes)),'macro_rmse':float(np.mean(rmses)),'target_cells':len(maes),'query_labels_used_for_training':False}); print(f'MAML seed={seed}',flush=True)
    # UDA: all target curves are unlabeled domain observations. Keep methods in
    # a separate track; they are never mixed into strict transfer ranking.
    for method in ['MMD','DeepCORAL','DANN']:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); model=Net().to(device); domain=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,2)).to(device) if method=='DANN' else None; params=list(model.parameters())+(list(domain.parameters()) if domain else []); opt=torch.optim.AdamW(params,lr=2e-3,weight_decay=1e-4); g=torch.Generator().manual_seed(seed); xs=torch.from_numpy(sx).to(device);ys=torch.from_numpy(sy[:,None]).float().to(device); xt=torch.from_numpy(unlabeled).to(device); n=min(len(xs),len(xt))
            for _ in range(20):
                si=torch.randperm(len(xs),generator=g)[:n];ti=torch.randperm(len(xt),generator=g)[:n];fs=model.feat(xs[si]);ft=model.feat(xt[ti]);loss=nn.functional.mse_loss(model.head(fs),ys[si])
                if method=='MMD': loss=loss+.1*((fs.mean(0)-ft.mean(0))**2).mean()
                elif method=='DeepCORAL':
                    a=fs-fs.mean(0,keepdim=True);b=ft-ft.mean(0,keepdim=True); loss=loss+.1*(((a.T@a)/max(1,len(a)-1)-(b.T@b)/max(1,len(b)-1))**2).mean()
                else:
                    dom=torch.cat([fs,ft]); dy=torch.cat([torch.zeros(len(fs),dtype=torch.long,device=device),torch.ones(len(ft),dtype=torch.long,device=device)]);loss=loss+.1*nn.functional.cross_entropy(domain(GRL.apply(dom,1.0)),dy)
                opt.zero_grad();loss.backward();opt.step()
            maes=[];rmses=[]
            for r in te:
                with torch.no_grad():p=model(torch.from_numpy(r['x'][args.K:]).to(device)).cpu().numpy().ravel()
                y=r['y'][args.K:];mae=float(np.mean(abs(p-y)));rmse=float(np.sqrt(np.mean((p-y)**2)));maes.append(mae);rmses.append(rmse);cells.append({'model':method,'seed':seed,'K':args.K,'cell_id':r['id'],'mae':mae,'rmse':rmse})
            rows.append({'model':method,'seed':seed,'K':args.K,'macro_mae':float(np.mean(maes)),'macro_rmse':float(np.mean(rmses)),'target_cells':len(maes),'query_labels_used_for_training':False}); print(f'{method} seed={seed}',flush=True)
    for fn,data in [('transfer_results.csv',rows),('transfer_cell_results.csv',cells)]:
        with (out/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    man={'status':'CORRECTED_TRANSFER_AND_UDA_COMPLETE','task':'XJTU_to_HUST','K':args.K,'seeds':seeds,'methods':['MAML','MMD','DeepCORAL','DANN'],'source_cache':str(src.relative_to(ROOT)),'target_cache':str(tgt.relative_to(ROOT)),'source_selection_sha256':digest(ROOT/'实验部署/source_window_cache_v2/window_selection.json'),'target_cache_manifest_sha256':digest(tgt/'manifest.json'),'target_soh_labels_used_for_uda_training':False,'uda_separate_track':True}
    (out/'manifest.json').write_text(json.dumps(man,indent=2))
if __name__=='__main__':main()
