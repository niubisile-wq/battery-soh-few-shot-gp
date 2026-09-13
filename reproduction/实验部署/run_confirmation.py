"""Frozen final-model confirmation; no source fitting or target-based selection."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
from study import HERE,BUNDLE,PROTOCOL_PATH,digest,write_json,write_csv,preservation

sys.path.insert(0,str(BUNDLE/'source_snapshot/m4'))
import random_function_module
from random_function_gp import RandomFunctionGP
from source_oof import sa
from data import Cell
from sklearn.gaussian_process import GaussianProcessRegressor
from threadpoolctl import threadpool_limits

OUT=HERE/'external/confirmation'
GROUPS=['B','B1','B12','B123','B1234']


def forbidden_fit(*args,**kwargs):
    raise AssertionError('No core/source model fitting is permitted in frozen confirmation')


def load_sources(fold,lock):
    adapters={}
    for group in ['B4','B14','B124','B1234']:
        path=BUNDLE/'folds'/fold.replace(':','__')/group/'adapter.joblib'
        assert digest(path)==lock['source_models'][str(path)]
        adapters[group]=joblib.load(path)
    return adapters


def predict(adapters,group,cell):
    key={'B':'B4','B1':'B14','B12':'B124','B123':'B1234','B1234':'B1234'}[group]
    model=adapters[key]
    if group=='B1234':p=model.predict(cell)
    else:p=model.parent.predict(cell) if model.parent_mode is None else model.parent.predict(cell,model.parent_mode)
    p=np.asarray(p,dtype=float)
    assert p.shape==(len(cell.x)-10,) and np.isfinite(p).all()
    return p


def cell_from_spec(spec):
    assert digest(spec['cache'])==spec['cache_sha256']
    with np.load(spec['cache'],allow_pickle=False) as z:
        x=z['x'].astype('float32');q=z['capacity_Ah'].astype('float32');cycle=z['cycle_number'].astype('float32')
    y=q/q[0]
    hidden=Cell(spec['cell_id'],'CALCE_CONFIRMATION',spec['domain'],x,
        np.r_[y[:10],np.full(len(y)-10,np.nan)].astype('float32'),cycle,float(q[0]),spec['nominal_capacity_Ah'],spec['cache'])
    return hidden,y


def run_fold(fold):
    request=json.loads((OUT/'request.json').read_text());lock=json.loads((HERE/'external/exposure_and_source_lock.json').read_text())
    assert request['code_sha256']==digest(Path(__file__))
    dest=OUT/'folds'/fold.replace(':','__');dest.mkdir(parents=True,exist_ok=True)
    complete=dest/'complete.json'
    if complete.exists():
        done=json.loads(complete.read_text());assert done['request_sha256']==digest(OUT/'request.json')
        for p,h in done['artifacts'].items():assert digest(p)==h
        return done
    sa.GPModel.fit=forbidden_fit;GaussianProcessRegressor.fit=forbidden_fit;RandomFunctionGP.fit=forbidden_fit
    start=time.monotonic();rows=[];boundaries=[];artifacts=[]
    with threadpool_limits(limits=1):
        adapters=load_sources(fold,lock)
        for spec in request['cells']:
            cell,y=cell_from_spec(spec);assert np.isnan(cell.y[10:]).all()
            for group in GROUPS:
                p=predict(adapters,group,cell)
                altered=replace(cell,y=np.r_[y[:10],np.random.default_rng(91010).uniform(-3,5,len(y)-10)].astype('float32'))
                pp=predict(adapters,group,altered)
                label_delta=float(np.max(abs(pp-p)));assert label_delta<1e-8
                checks=[]
                for j in sorted({10,(len(y)+10)//2,len(y)-1}):
                    ix=np.r_[np.arange(10),j]
                    local=replace(cell,x=cell.x[ix],y=cell.y[ix],cycle=cell.cycle[ix])
                    checks.append(abs(float(predict(adapters,group,local)[0])-float(p[j-10])))
                current_delta=float(max(checks));assert current_delta<1e-7,(fold,cell.id,group,current_delta)
                path=dest/'predictions'/cell.id/(group+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,pred=p,y=y[10:],cycle=cell.cycle[10:],support_y=y[:10])
                artifacts.append(path);e=(p-y[10:].astype(float))*100;a=abs(e);low=y[10:].astype(float)<.9
                rows.append(dict(source_fold=fold,cell_id=cell.id,domain=cell.domain,group=group,query=len(e),
                    mae_pp=float(a.mean()),rmse_pp=float(np.sqrt(np.mean(e*e))),p95_ae_pp=float(np.quantile(a,.95)),
                    bias_pp=float(e.mean()),low_soh_query=int(low.sum()),low_soh_mae_pp=float(a[low].mean()) if low.any() else None,
                    prediction_file=str(path),source_selection='all_six_folds_no_target_based_selection'))
                boundaries.append(dict(source_fold=fold,cell_id=cell.id,group=group,query_labels_hidden=True,
                    altered_query_labels_max_abs=label_delta,current_query_points=len(checks),current_query_max_abs=current_delta,
                    source_fitting_disabled=True))
            print('CONFIRMED',fold,cell.id,'groups',len(GROUPS),flush=True)
    write_csv(dest/'metrics.csv',rows);write_json(dest/'boundary.json',boundaries);artifacts += [dest/'metrics.csv',dest/'boundary.json']
    for p,h in lock['source_models'].items():assert digest(p)==h
    done=dict(status='FROZEN_INFERENCE_COMPLETE',source_fold=fold,cells=len(request['cells']),groups=GROUPS,rows=len(rows),
        source_training_performed=False,target_hyperparameters_selected=False,request_sha256=digest(OUT/'request.json'),
        artifacts={str(p):digest(p) for p in artifacts},elapsed_seconds=time.monotonic()-start)
    write_json(complete,done);return done


def main():
    preservation();gate_path=HERE/'external/confirmation_data_gate.json';gate=json.loads(gate_path.read_text())
    assert gate['status']=='APPROVED_FOR_FROZEN_PREDICTION' and gate['all_pairs_rebuilt']
    for p,h in gate['input_hashes'].items():assert digest(p)==h
    lock_path=HERE/'external/exposure_and_source_lock.json';lock=json.loads(lock_path.read_text())
    for p,h in lock['source_models'].items():assert digest(p)==h
    folds=sorted({r['fold'] for r in json.loads((BUNDLE/'candidate.json').read_text())['manifest'] if r['fold'].startswith('XJTU:')})
    assert len(folds)==6
    request=dict(status='LOCKED_BEFORE_FIRST_NEW_COHORT_PREDICTION',created_unix=time.time(),cells=gate['eligible_cells'],source_folds=folds,groups=GROUPS,
        gate_sha256=digest(gate_path),source_lock_sha256=digest(lock_path),code_sha256=digest(Path(__file__)),
        protocol_sha256=digest(PROTOCOL_PATH),pl_protocol_sha256=digest(HERE/'external/pl_protocol.json'),
        parent_mapping={'B':'B4.parent','B1':'B14.parent','B12':'B124.parent','B123':'B1234.parent','B1234':'B1234'},
        no_source_training=True,no_target_selection=True,no_source_ensemble=True,
        aggregation='Report every source-fold and every cell; primary cohort descriptive mean: equal source folds, equal protocol groups, equal cells within group; do not treat folds or cycles as independent cells',
        limits=gate['scope'])
    if (OUT/'request.json').exists():
        old=json.loads((OUT/'request.json').read_text());assert {k:v for k,v in old.items() if k!='created_unix'}=={k:v for k,v in request.items() if k!='created_unix'}
    else:write_json(OUT/'request.json',request)
    done=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(run_fold,f) for f in folds]):
            result=future.result();done.append(result)
            write_json(OUT/'progress.json',dict(complete=done,total=6));print('FOLD_COMPLETE',len(done),'/6',result['source_fold'],flush=True)
    write_json(OUT/'inference_complete.json',dict(status='ALL_FROZEN_PREDICTIONS_SAVED_AUDIT_PENDING',folds=done,
        request_sha256=digest(OUT/'request.json'),total_cell_group_fold_rows=sum(r['rows'] for r in done)))


if __name__=='__main__':main()
