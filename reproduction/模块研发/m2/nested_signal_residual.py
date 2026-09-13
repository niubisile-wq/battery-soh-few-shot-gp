"""Nested source-only signed residual learning from current signal changes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits
import nested_signal_probe as ns


def kernel(x,z,anchored):
    gamma=1/x.shape[1]
    k=rbf_kernel(x,z,gamma=gamma)
    if anchored:
        k=k-np.exp(-gamma*np.sum(x*x,axis=1))[:,None]-np.exp(-gamma*np.sum(z*z,axis=1))[None,:]+1
    return k


def fit(train,views,anchored):
    weights=ns.balanced_weights(train)
    x=np.concatenate([views[e['cell_id']]['delta'] for e in train])
    y=np.concatenate([e['y']-e['base'] for e in train])
    keep=x.std(0)>ns.FLOOR
    if not keep.any():return dict(degenerate=True)
    scaler=StandardScaler(with_mean=False).fit(x[:,keep],sample_weight=weights)
    z=scaler.transform(x[:,keep])
    model=KernelRidge(alpha=.01,kernel='precomputed').fit(kernel(z,z,anchored),y,sample_weight=weights)
    return dict(keep=keep,scaler=scaler,z=z,anchored=anchored,model=model)


def correction(delta,head):
    if head.get('degenerate',False):return np.zeros(len(delta))
    x=head['scaler'].transform(delta[:,head['keep']])
    return head['model'].predict(kernel(x,head['z'],head['anchored']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=ns.rd.sa
    root=sa.ROOT/'模块研发/results/m2_nested_signal_probe_v1'
    audit=json.loads((root/'audit.json').read_text());assert audit['status']=='PASS'
    protocol=json.loads((root/'protocol.json').read_text());cells=sa.load_cells('MATR');rows=[];manifest=[]
    for fold in protocol['folds']:
        train,_,_=sa.split(cells,fold);allowed=sa.source_indices(train)
        views={c.id:ns.signal_views(sa.source_view(c,allowed[c.id]),allowed[c.id][allowed[c.id]>=sa.K]) for c in train}
        folder=root/'folds'/fold.replace(':','__');result=json.loads((folder/'result.json').read_text())
        pairs={}
        for excluded in sorted({tuple(f['excluded']) for f in result['fits']}):
            artifact=next(f['artifact'] for f in result['fits'] if tuple(f['excluded'])==excluded)
            pairs[excluded]=joblib.load((root/artifact).parent/'episodes.joblib')
        original=joblib.load(folder/'source_lodo_episodes.joblib')
        for held in sorted({c.domain for c in train}):
            for parent in ['Base','Base+M1']:
                records=[]
                for domain in sorted({c.domain for c in train}-{held}):
                    records.extend(e for e in pairs[tuple(sorted([held,domain]))][parent] if e['domain']==domain)
                validation=[e for e in original[parent] if e['domain']==held]
                assert not {e['cell_id'] for e in records}&{e['cell_id'] for e in validation}
                for method in ['unchanged','rbf_residual','anchored_residual']:
                    head=None if method=='unchanged' else fit(records,views,method=='anchored_residual')
                    if head is not None:
                        path=out/'heads'/fold.replace(':','__')/f'{held}__{parent}__{method}.joblib';path.parent.mkdir(parents=True,exist_ok=True)
                        joblib.dump(head,path,compress=3)
                        manifest.append(dict(fold=fold,held=held,parent=parent,method=method,artifact=str(path.relative_to(out)),sha256=sa.digest(path),
                            train_ids=[e['cell_id'] for e in records],validation_ids=[e['cell_id'] for e in validation]))
                    for e in validation:
                        p=e['base'] if head is None else e['base']+correction(views[e['cell_id']]['delta'],head)
                        m=sa.metrics(e['y'],p)
                        rows.append(dict(fold=fold,domain=held,parent=parent,method=method,cell_id=e['cell_id'],**{k:float(m[k]*100) for k in ['mae','rmse','p95_ae']}))
    summary=[]
    for parent in ['Base','Base+M1']:
        for method in ['unchanged','rbf_residual','anchored_residual']:
            grouped=defaultdict(list)
            for r in rows:
                if r['parent']==parent and r['method']==method:grouped[r['fold'],r['domain']].append(r)
            summary.append(dict(parent=parent,method=method,**{k:float(np.mean([np.mean([r[k] for r in g]) for g in grouped.values()])) for k in ['mae','rmse','p95_ae']}))
    sa.write_json(out/'result.json',dict(status='COMPLETE',summary=summary,rows=rows,manifest=manifest,
        parameters=dict(alpha=.01,gamma='1/resolved feature count',weights='equal domains, cells, queries'),
        limits=['MATR source-only nested GP fitting; original fixed configurations inherited, not nested configuration selection.',
                'Current HI minus support HI mean; no query labels or future signals at inference.',
                'Anchored kernel gives zero correction at support-mean features, not at every individual support cycle.',
                'Signed residual may extrapolate beyond the expert convex hull, but improvement is not guaranteed.',
                'Repeated development diagnostic, not full M2 or outer test results.']))
    for p in [Path(__file__),Path(ns.__file__)]:shutil.copyfile(p,out/p.name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
