"""Reconstruct selected source encoders and GP training arrays from raw budget rows."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import cho_solve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=HERE.parent/'results';candidate=root/'m3_waveform_candidate_v1'
    obj=json.loads((candidate/'candidate.json').read_text());old=json.loads((FROZEN/'candidate.json').read_text())
    sa.write_json(out/'protocol.json',dict(code_sha256=sa.digest(Path(__file__)),candidate_sha256=sa.digest(candidate/'candidate.json'),
        scope='All21 selected branches: independently build waveform rows, refit scaler/PCA, reconstruct GP normalized labels and posterior algebra. No optimizer restart or model changes.'))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];byid={c.id:c for c in cells};audits=[]
    with threadpool_limits(limits=1):
        for fold in sorted({r['fold'] for r in obj['manifest']}):
            entry=next(r for r in obj['manifest'] if r['fold']==fold)
            path=Path(entry['branch_source']);assert sa.digest(path)==entry['branch_sha256']
            model=joblib.load(path).parent;gp=model.gp
            train,val,test=sa.split(cells,fold);train_ids={c.id for c in train}
            keys=[tuple(k) for k in model.source_keys];allowed=defaultdict(list)
            for cid,j in keys:allowed[cid].append(j)
            assert len(keys)==len(set(keys))<=1000 and set(allowed)==train_ids
            assert train_ids.isdisjoint({c.id for c in val+test})
            original=next(r for r in old['manifest'] if r['fold']==fold)
            assert set(keys)=={tuple(k) for k in original['source_keys']}
            features={};targets={}
            for cid,indices in allowed.items():
                c=byid[cid];assert set(range(sa.K))<=set(indices)
                # Independent implementation, not waveform_features().
                voltage=np.asarray(c.x[:,0],dtype=float)-3.5
                current=np.asarray(c.x[:,1],dtype=float)/c.reference_capacity
                time=np.asarray(c.x[:,2],dtype=float);time=(time-time[:,:1])/3600.
                raw=np.stack([voltage,current,time],axis=1).reshape(len(c.x),-1)
                features[cid]=np.concatenate([raw,raw-raw[:sa.K].mean(0)],axis=1)
                targets[cid]=c.y-np.mean(c.y[:sa.K])
            x=np.stack([features[cid][j] for cid,j in keys]);y=np.array([targets[cid][j] for cid,j in keys])
            floor=np.tile(np.repeat([1e-6,1e-4,1e-7],128),2)
            keep=np.flatnonzero(x.std(0)>floor);assert np.array_equal(keep,model.keep)
            raw_scaler=StandardScaler().fit(x[:,keep]);z=raw_scaler.transform(x[:,keep])
            projection=PCA(n_components=min(8,z.shape[1],len(z)-1),svd_solver='full').fit(z)
            latent=projection.transform(z);lk=np.flatnonzero(latent.std(0)>1e-6)
            assert np.array_equal(lk,model.latent_keep)
            scaler=StandardScaler().fit(latent[:,lk]);xx=scaler.transform(latent[:,lk])
            errors=dict(raw_mean=float(np.max(abs(raw_scaler.mean_-model.raw_scaler.mean_))),
                raw_scale=float(np.max(abs(raw_scaler.scale_-model.raw_scaler.scale_))),
                pca=float(np.max(abs(projection.components_-model.projection.components_))),
                latent_mean=float(np.max(abs(scaler.mean_-model.scaler.mean_))),
                latent_scale=float(np.max(abs(scaler.scale_-model.scaler.scale_))),
                gp_input=float(np.max(abs(xx-gp.X_train_))))
            ym=y.mean();ys=y.std();ys=1. if ys==0 else ys
            errors['gp_target']=float(np.max(abs((y-ym)/ys-gp.y_train_)))
            errors['target_mean']=float(abs(ym-gp._y_train_mean));errors['target_scale']=float(abs(ys-gp._y_train_std))
            k=gp.kernel_(gp.X_train_);k[np.diag_indices_from(k)]+=gp.alpha
            errors['cholesky']=float(np.max(abs(k-gp.L_@gp.L_.T)))
            alpha=cho_solve((gp.L_,True),gp.y_train_)
            errors['posterior_alpha']=float(np.max(abs(alpha-gp.alpha_)))
            assert max(errors.values())<1e-8,errors
            audits.append(dict(fold=fold,source_rows=len(keys),source_cells=len(train),validation_cells=len(val),test_cells=len(test),errors=errors))
            print(fold,'source encoder and GP arrays reconstructed',max(errors.values()),flush=True)
    assert len(audits)==21 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in old['manifest'])
    sa.write_json(out/'audit.json',dict(status='PASS',folds=audits,
        limits='Verifies actual source arrays and GP algebra, not global optimum of kernel optimization or novelty.'))


if __name__=='__main__':main()
