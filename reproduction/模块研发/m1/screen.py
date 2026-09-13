"""Resumable, auditable development screens with inner-only configuration choice."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import hashlib
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from data import HERE,ROOT,OUT,K,PROTOCOL,load_cells,split,source_indices,provenance,write_json,digest,metrics,natural
from gp import GPModel

SPECS=json.loads((HERE/'candidates_v1.json').read_text())
CELLS=None


def init_worker(datasets):
    global CELLS
    CELLS=[c for ds in datasets for c in load_cells(ds)]


def macro(rows,key='mae'):
    groups=defaultdict(list)
    for r in rows:
        if r[key] is not None:groups[(r['dataset'],r['domain'])].append(r[key])
    datasets=defaultdict(list)
    for (ds,_),values in groups.items():datasets[ds].append(float(np.mean(values)))
    return float(np.mean([np.mean(v) for v in datasets.values()])) if datasets else None


def signature(spec):
    obj={'spec':spec,'protocol':digest(HERE/'protocol.json'),
         'code':{name:digest(HERE/name) for name in ['data.py','features.py','gp.py','screen.py']}}
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def run_one(item):
    name,fold,out_name=item
    spec=SPECS[name];sig=signature(spec)
    job=OUT/out_name/'jobs'/f'{name}__{fold.replace(":","__")}'
    if (job/'complete.json').exists():
        r=json.loads((job/'complete.json').read_text())
        if r['signature']!=sig:raise RuntimeError(f'Immutable result has different source signature: {job}')
        return r
    started=time.time();job.mkdir(parents=True,exist_ok=True)
    try:
        tr,va,te=split(CELLS,fold);idx=source_indices(tr)
        write_json(job/'provenance.json',{**provenance(tr,va,te,idx),'candidate':name,'spec':spec,'fold':fold,'signature':sig})
        trials=[];models=[]
        for ci,cfg in enumerate(spec['configs']):
            model=GPModel(spec,cfg).fit(tr,idx);models.append(model)
            for mode in spec['modes']:
                rows=[]
                for c in va:
                    pred=model.predict(c,mode)
                    rows.append({'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,**metrics(c.y[K:],pred)})
                trials.append({'configuration':ci,'config':cfg,'mode':mode,'inner_mae':macro(rows),
                               'inner_rmse':macro(rows,'rmse'),'validation_cells':rows,'source_fit':model.serialize()})
            write_json(job/'tuning.json',{'status':'TUNING','trials':trials})
        chosen=min(trials,key=lambda r:r['inner_mae'])
        model=models[chosen['configuration']]
        write_json(job/'tuning.json',{'status':'CHOSEN_ON_INNER_VALIDATION','chosen':chosen,'trials':trials})
        joblib.dump(model,job/'model.joblib',compress=3)
        rows=[];max_difference=0.
        (job/'predictions').mkdir(exist_ok=True)
        for c in te:
            pred=model.predict(c,chosen['mode'])
            path=job/'predictions'/f'{Path(c.id).stem}.npz'
            np.savez_compressed(path,y=c.y[K:],pred=pred,cycle=c.cycle[K:])
            row={'candidate':name,'fold':fold,'dataset':c.dataset,'domain':c.domain,'cell_id':c.id,
                 'mode':chosen['mode'],'configuration':chosen['configuration'],**metrics(c.y[K:],pred),
                 'prediction_file':str(path.relative_to(ROOT))}
            if name=='C00_original' and fold.startswith('XJTU:'):
                old=ROOT/'开发基线选择依据/results/fair_selection_v1/jobs'/f'GPR__{c.domain}__s0'/'predictions'/f'GPR__source_only__s0__{Path(c.id).stem}.npz'
                with np.load(old) as z:
                    if not np.array_equal(z['y'],c.y[K:]):raise AssertionError('Champion labels differ')
                    diff=float(np.max(abs(z['pred']-pred)))
                max_difference=max(max_difference,diff)
                if diff>1e-10:raise AssertionError(f'Champion reproduction mismatch {diff}')
            rows.append(row)
        write_json(job/'cell_results.json',rows)
        r={'status':'COMPLETE','candidate':name,'fold':fold,'signature':sig,'cells':len(rows),
           'mae':macro(rows),'rmse':macro(rows,'rmse'),'p95_ae':macro(rows,'p95_ae'),
           'mode':chosen['mode'],'elapsed_s':time.time()-started,
           'champion_max_prediction_difference':max_difference if name=='C00_original' and fold.startswith('XJTU:') else None}
        write_json(job/'complete.json',r)
        return r
    except Exception:
        write_json(job/'failure.json',{'status':'FAILED','candidate':name,'fold':fold,'signature':sig,'traceback':traceback.format_exc()})
        raise


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--datasets',default='XJTU')
    ap.add_argument('--candidates',default=','.join(SPECS))
    ap.add_argument('--folds',default='all')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--out-name',default='screen_v1')
    args=ap.parse_args();datasets=args.datasets.split(',')
    info=[c for ds in datasets for c in load_cells(ds)]
    folds=sorted({f'{c.dataset}:{c.domain}' for c in info}) if args.folds=='all' else args.folds.split(',')
    names=args.candidates.split(',')
    for name in names:
        if name not in SPECS:raise ValueError(name)
    specs=[(name,f,args.out_name) for name in names for f in folds]
    manifest={'status':'REQUESTED','datasets':datasets,'folds':folds,'candidates':names,
              'expected_jobs':len(specs),'specifications':{n:SPECS[n] for n in names},
              'protocol':PROTOCOL,'signatures':{n:signature(SPECS[n]) for n in names}}
    batch=OUT/args.out_name/'batches'/f'{time.time_ns()}.json'
    write_json(batch,manifest)
    failures=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(datasets,)) as pool:
        future={pool.submit(run_one,s):s for s in specs}
        for f in as_completed(future):
            try:print(json.dumps(f.result()),flush=True)
            except Exception as e:
                failures.append({'job':future[f],'error':repr(e)})
                print(json.dumps({'status':'FAILED',**failures[-1]}),flush=True)
    manifest.update(status='COMPLETE' if not failures else 'INCOMPLETE',failures=failures)
    write_json(batch,manifest)
    if failures:raise SystemExit(1)
    print('REQUESTED_SCREEN_COMPLETE',flush=True)


if __name__=='__main__':main()
