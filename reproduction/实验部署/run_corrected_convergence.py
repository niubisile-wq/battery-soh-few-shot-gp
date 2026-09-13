"""Source-development convergence evidence on corrected measured SOH labels.

All checkpoints use source-validation only. No target archive is opened.
Per-epoch evidence and per-run checkpoints are flushed before the next run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import run_formal_protocol as formal

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    base=ROOT/'实验部署/source_window_cache_v2'
    selection=json.loads((base/'window_selection.json').read_text())
    if selection['status']!='WINDOW_SELECTED_ON_SOURCE_VALIDATION_ONLY':
        raise RuntimeError('Window not selected')
    data={}
    for split in ['source_train','source_validation']:
        data[split]=[]
        for path in sorted((base/selection['selected']/split).glob('*.npz')):
            with np.load(path) as a:
                q=a['capacity_Ah'];x=a['x'];n=a['cycle_number']
                if len(q)<=10 or not .5<=q[0]/float(a['nominal_capacity_Ah'])<=1.5:
                    raise RuntimeError(f'Unresolved reference or insufficient support: {path}')
                data[split].append((str(a['cell_id']),x.astype('float32'),(q/q[0]).astype('float32'),n))
    # Source train budget fixed cell by cell, avoiding protocol ordering bias.
    tr=data['source_train'];budget=1000
    xs=[];ys=[];sample_keys=[]
    for index,(cid,x,y,n) in enumerate(tr):
        quota=budget//len(tr)+int(index<budget%len(tr))
        keep=np.linspace(0,len(y)-1,min(quota,len(y))).astype(int)
        xs.append(x[keep]);ys.append(y[keep]);sample_keys.extend((cid,float(n[k])) for k in keep)
    x=np.concatenate(xs);y=np.concatenate(ys)
    mu=x.mean((0,2),keepdims=True);sd=x.std((0,2),keepdims=True);sd[sd<1e-6]=1
    norm=lambda a:((a-mu)/sd).astype('float32')
    val=[(cid,norm(a[10:]),b[10:]) for cid,a,b,_ in data['source_validation']]
    return norm(x),y,val,{'selected_window':selection['selected'],
         'window_selection_sha256':digest(base/'window_selection.json'),
         'source_samples':sample_keys,'source_train_cells':[c[0] for c in tr],
         'source_validation_cells':[c[0] for c in val],
         'normalization_mean':mu.tolist(),'normalization_std':sd.tolist(),
         'label_policy':'first reliable measured reference capacity; never partial-discharge quantity',
         'validation_query':'after first 10 eligible reference cycles'}


def evaluate(model,val,device):
    model.eval();ma=[];ms=[]
    with torch.no_grad():
        for cid,x,y in val:
            preds=[]
            for start in range(0,len(x),256):
                preds.append(model(torch.from_numpy(x[start:start+256]).to(device)).cpu().numpy().ravel())
            p=np.concatenate(preds)
            if not np.isfinite(p).all():raise RuntimeError('Nonfinite validation predictions')
            ma.append(float(np.mean(abs(p-y))));ms.append(float(np.mean((p-y)**2)))
    return float(np.mean(ma)),float(np.mean(ms))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=150)
    ap.add_argument('--seeds',default='0,1,2');ap.add_argument('--models',default=','.join(formal.NN_MODELS))
    ap.add_argument('--output',default='corrected_convergence_v1');args=ap.parse_args()
    out=ROOT/'开发基线选择依据/results'/args.output;out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Use a new output directory; do not overwrite run evidence')
    x,y,val,manifest=load_source();torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest.update(arguments=vars(args),device=str(device),batch_size=64,learning_rate=.002,
                    target_archives_read=[],status='RUNNING',
                    runner_sha256=digest(Path(__file__)),
                    model_code_sha256=digest(Path(formal.__file__)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]
    for name in args.models.split(','):
        for seed in map(int,args.seeds.split(',')):
            formal.seed(seed);model=formal.NN_MODELS[name]().to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
            loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y[:,None])),batch_size=64,shuffle=True)
            best=float('inf');best_epoch=0;updates=0;state=None;history=[]
            with (out/f'{name}_seed{seed}_curve.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['epoch','optimizer_updates','train_mse','validation_macro_mae','validation_macro_mse']);writer.writeheader()
                for epoch in range(1,args.epochs+1):
                    model.train();total=0.
                    for xb,yb in loader:
                        xb=xb.to(device);yb=yb.to(device);optimizer.zero_grad(set_to_none=True)
                        loss=torch.nn.functional.mse_loss(model(xb),yb)
                        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {name} {seed}')
                        loss.backward();optimizer.step();updates+=1;total+=float(loss.detach())*len(xb)
                    mae,mse=evaluate(model,val,device)
                    row=dict(epoch=epoch,optimizer_updates=updates,train_mse=total/len(x),validation_macro_mae=mae,validation_macro_mse=mse)
                    writer.writerow(row);f.flush();history.append(row)
                    if mse<best:
                        best=mse;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if epoch%25==0:print(f'{name} seed={seed} epoch={epoch} updates={updates} val_mae={mae:.5f} best_epoch={best_epoch}',flush=True)
            torch.save({'state_dict':state,'model':name,'seed':seed,'epoch':best_epoch,'manifest':manifest},out/f'{name}_seed{seed}_best.pt')
            first=min(r['validation_macro_mse'] for r in history[:25])
            records.append(dict(model=name,seed=seed,best_epoch=best_epoch,best_validation_mse=best,
                                relative_improvement_after_epoch25=(first-best)/max(first,1e-12),
                                optimizer_updates=updates,late_best=best_epoch>args.epochs-25))
            (out/'run_summary.json').write_text(json.dumps(records,indent=2))
    manifest['status']='CONVERGENCE_DIAGNOSTICS_COMPLETE_NOT_READINESS_APPROVAL'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
