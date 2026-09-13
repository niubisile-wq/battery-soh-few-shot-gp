"""Matched, source-validation-selected baselines. All writes are study-local."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from study import HERE, ROOT, BUNDLE, PROTOCOL, PROTOCOL_PATH, digest, write_json, write_csv, preservation

FAIR = ROOT/'开发基线选择依据/fair_selection'
sys.path.insert(0, str(ROOT/'模块研发/m1'))
import data
sys.path.insert(0, str(FAIR))
import common
import neural
import transfer
import numpy as np
import torch
from threadpoolctl import threadpool_limits

OUT = HERE/'baselines'
K = PROTOCOL['K']
OLD = ROOT/'开发基线选择依据/results/fair_selection_v1'


def samples(cells):
    cells = sorted(cells, key=lambda c:data.natural(c.id))
    idx = data.source_indices(cells)
    x = np.concatenate([c.x[idx[c.id]] for c in cells])
    y = np.concatenate([c.y[idx[c.id]] for c in cells])
    t = np.concatenate([c.cycle[idx[c.id]] for c in cells])
    groups = np.concatenate([np.full(len(idx[c.id]), i) for i,c in enumerate(cells)])
    keys = [[c.id, float(c.cycle[j])] for c in cells for j in idx[c.id]]
    assert len(keys) <= 1000 and len(cells) <= 60
    return x, y, t, groups, keys


def sha_files(paths):
    return {str(p):digest(p) for p in paths}


def validate_reuse(dataset, domain, name, seed, tr, va, te, keys, norm):
    if dataset != 'XJTU':
        return None
    job = OLD/'jobs'/f'{name}__{domain}__s{seed}'
    p = json.loads((job/'provenance.json').read_text())
    assert set(p['train_cells']) == {c.id for c in tr}
    assert set(p['tune_cells']) == {c.id for c in va}
    assert set(p['development_target_cells']) == {c.id for c in te}
    assert {tuple(k) for k in p['source_sample_keys']} == {tuple(k) for k in keys}
    for key in ['mean','std','source_cycle_scale']:
        np.testing.assert_allclose(p['normalization'][key], norm.serialize()[key], rtol=0, atol=1e-7)
    assert p['protocol_sha256'] == digest(FAIR/'protocol.json')
    for f in ['common.py','neural.py','transfer.py','worker.py']:
        assert digest(FAIR/f) == p['code_sha256'][f], ('changed training/inference helper',f)
    training = json.loads((job/'training.json').read_text())
    complete = json.loads((job/'complete.json').read_text())
    assert complete['status'] == 'COMPLETE'
    return dict(checkpoint=job/'source.pt', training=training,
                hashes=sha_files([job/'source.pt',job/'training.json',job/'provenance.json',job/'complete.json']))


def validation_choices(base, name, validation, seed, saved=None):
    if saved is None:
        options, trials = neural.choose_adaptation(base, name, validation, seed, 'cpu')
    else:
        options, trials = saved['adaptation'], saved['adaptation_trials']
    candidates = []
    for mode in ['source_only','source_bias']:
        scores = []
        for x,y in validation:
            p = neural.predict(base,x)
            q = p[K:] if mode == 'source_only' else p[K:]+(y[:K]-p[:K]).mean()
            scores.append(float(abs(q-y[K:]).mean()))
        candidates.append(dict(mode=mode, inner_mae=float(np.mean(scores))))
    for mode, option in options.items():
        candidates.append(dict(option, mode=mode))
    choice = min(candidates, key=lambda r:r['inner_mae'])
    return dict(primary=choice, adaptation=options, trials=trials, mode_scores=candidates)


def fresh_source(name, tr, norm, x, y, groups, val, seed, tuning, attempt, chosen=None):
    maml = name == 'MAML_MLP'
    architecture = 'MLP' if maml else name
    transfer.sample_source = samples
    if seed == 0:
        candidates = []
        for lr in ([.0003,.001] if maml else common.PROTOCOL['neural']['learning_rates']):
            log = attempt/f'lr{lr}_curve.csv'
            print('FIT',name,'seed',seed,'lr',lr,flush=True)
            if maml:
                model, info = transfer.maml_train(tr,norm,val,lr,seed,log,'cpu')
            else:
                model, info = neural.train_source(name,norm.signal(x),y,[(a[K:],b[K:]) for a,b in val],
                    lr,seed,log,'cpu',config=common.PROTOCOL['neural'],groups=groups)
            checkpoint = attempt/f'lr{lr}.pt'
            torch.save(model.state_dict(),checkpoint)
            candidates.append(dict(**info,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint)))
        selected = min(candidates,key=lambda r:r['best_inner_mae'])
        model = neural.MODELS[architecture]().to('cpu')
        model.load_state_dict(torch.load(selected['checkpoint'],weights_only=True,map_location='cpu'))
        return model, selected, candidates
    lr = chosen['pretrain']['lr']
    if maml:
        model,info = transfer.maml_train(tr,norm,val,lr,seed,attempt/'source_curve.csv','cpu')
    else:
        model,info = neural.train_source(name,norm.signal(x),y,[(a[K:],b[K:]) for a,b in val],
            lr,seed,attempt/'source_curve.csv','cpu',config=common.PROTOCOL['neural'],groups=groups)
    return model,info,[]


def infer(base, name, norm, visible, choice, seed):
    """No reference to target query labels or to target validation scores."""
    assert np.isnan(visible.y[K:]).all()
    a = norm.signal(visible.x)
    model = base
    mode = choice['mode']
    source = neural.predict(base,a)
    if mode in ['source_only','source_bias']:
        p = source[K:].copy()
        if mode == 'source_bias':
            p += float((visible.y[:K]-source[:K]).mean())
        fit_log = dict(optimizer_updates=0)
    else:
        model,fit_log = neural.fit_support(base,'MLP' if name=='MAML_MLP' else name,
            a[:K],visible.y[:K],mode,choice['lr'],choice['steps'],seed+10000,'cpu')
        p = neural.predict(model,a[K:])
    return p, source, model, fit_log


def run_fold(dataset,domain,name):
    torch.set_num_threads(1)
    fold=f'{dataset}:{domain}'
    job=OUT/f'{dataset}__{domain}__{name}'
    job.mkdir(parents=True,exist_ok=True)
    cells=data.load_cells(dataset)
    tr,va,te=data.split(cells,fold)
    x,y,cycle,groups,keys=samples(tr)
    norm=common.Normalizer(x,cycle)
    # Bind to the final model's actual source universe, not just matching its count.
    manifest=json.loads((BUNDLE/'candidate.json').read_text())
    source=next(r for r in manifest['manifest'] if r['fold']==fold and r['group']=='B1234')
    indices=data.source_indices(tr)
    positional_keys={(c.id,int(j)) for c in tr for j in indices[c.id]}
    assert {tuple(k) for k in source['source_keys']} == positional_keys
    paths=[FAIR/'neural.py',FAIR/'transfer.py',FAIR/'common.py',FAIR/'protocol.json',
           FAIR.parent/'run_formal_protocol.py',Path(data.__file__),ROOT/'模块研发/m1/protocol.json',Path(__file__)]
    provenance=dict(dataset=dataset,domain=domain,model=name,
        train_cells=[c.id for c in tr],validation_cells=[c.id for c in va],target_cells=[c.id for c in te],
        source_sample_keys=keys,source_positional_keys=sorted(positional_keys),
        normalization=norm.serialize(),protocol_sha256=digest(PROTOCOL_PATH),
        data_sha256=sha_files([ROOT/c.path for c in tr+va+te]),code_sha256=sha_files(paths),
        final_source_universe_verified=True,torch_version=torch.__version__)
    provenance=json.loads(json.dumps(provenance))
    provenance_path=job/'provenance.json'
    if provenance_path.exists():
        prior=json.loads(provenance_path.read_text())
        if prior != provenance:
            assert not list(job.glob('seed_*/complete.json')) and not (job/'chosen.json').exists(), 'Do not mix executed study versions'
            assert {k:v for k,v in prior.items() if k!='code_sha256'} == {k:v for k,v in provenance.items() if k!='code_sha256'}
            write_json(job/'preflight_history'/f'{time.time_ns()}.json',prior)
            write_json(provenance_path,provenance)
    else:
        write_json(provenance_path,provenance)
    validation=[(norm.signal(c.x),c.y) for c in va]
    tuning=job/'chosen.json'
    for seed in PROTOCOL['baseline_seeds']:
        dest=job/f'seed_{seed}';dest.mkdir(exist_ok=True)
        complete=dest/'complete.json'
        if complete.exists():
            done=json.loads(complete.read_text())
            assert done['protocol_sha256']==digest(PROTOCOL_PATH)
            assert done['provenance_sha256']==digest(provenance_path)
            for p,h in done['artifacts'].items(): assert digest(p)==h
            print('REUSE_COMPLETED',dataset,domain,name,seed,flush=True)
            continue
        attempt=dest/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
        started=time.monotonic()
        chosen=json.loads(tuning.read_text()) if tuning.exists() else None
        old=validate_reuse(dataset,domain,name,seed,tr,va,te,keys,norm)
        if old:
            architecture='MLP' if name=='MAML_MLP' else name
            base=neural.MODELS[architecture]().to('cpu')
            base.load_state_dict(torch.load(old['checkpoint'],map_location='cpu',weights_only=True))
            info=old['training']['pretrain'];trials=[]
        else:
            base,info,trials=fresh_source(name,tr,norm,x,y,groups,validation,seed,tuning,attempt,chosen)
        if seed==0 and chosen is None:
            if name=='MAML_MLP':
                if old:
                    candidate=old['training']['chosen'];steps=candidate['adapt_steps'];at=candidate['adaptation_trials']
                else:
                    at=[]
                    for steps in common.PROTOCOL['neural']['adapt_steps']:
                        scores=[]
                        for a,b in validation:
                            fitted,_=neural.fit_support(base,'MLP',a[:K],b[:K],'full_finetune',.0005,steps,10000,'cpu')
                            scores.append(float(abs(neural.predict(fitted,a[K:])-b[K:]).mean()))
                        at.append(dict(steps=steps,inner_mae=float(np.mean(scores))))
                    steps=min(at,key=lambda r:r['inner_mae'])['steps']
                adaptation=dict(primary=dict(mode='full_finetune',lr=.0005,steps=steps),trials=at)
            else:
                adaptation=validation_choices(base,name,validation,0,old['training']['chosen'] if old else None)
            chosen=dict(pretrain=info,source_trials=trials,**adaptation,protocol_sha256=digest(PROTOCOL_PATH),
                        selection_scope='source validation only',reused=old['hashes'] if old else None)
            write_json(tuning,chosen)
        assert chosen is not None and chosen['protocol_sha256']==digest(PROTOCOL_PATH)
        source_path=dest/'source.pt';torch.save(base.state_dict(),source_path)
        rows=[];fits=[];artifacts=[source_path];boundary=[]
        for cell in te:
            visible=replace(cell,y=np.r_[cell.y[:K],np.full(len(cell.y)-K,np.nan)].astype('float32'))
            p,source_pred,fitted,fit_log=infer(base,name,norm,visible,chosen['primary'],seed)
            cell_dir=dest/'cells'/cell.id;cell_dir.mkdir(parents=True,exist_ok=True)
            torch.save(fitted.state_dict(),cell_dir/'adapted.pt');artifacts.append(cell_dir/'adapted.pt')
            predictions={'inner_selected':p,'source_only':source_pred[K:],
                         'source_bias':source_pred[K:]+float((cell.y[:K]-source_pred[:K]).mean())}
            # Pointwise replay tests current-query execution without future query signals.
            if chosen['primary']['mode'] not in ['source_only','source_bias']:
                prefix=neural.predict(fitted,norm.signal(cell.x[K:K+1]))
                difference=float(np.max(abs(prefix-p[:1])))
            else:
                prefix=neural.predict(base,norm.signal(cell.x[K:K+1]))
                difference=float(np.max(abs(prefix-source_pred[K:K+1])))
            assert difference<1e-5,('query prefix dependence',cell.id,difference)
            boundary.append(dict(cell_id=cell.id,query_labels_masked=True,current_query_replay_max_abs=difference))
            for setting,pred in predictions.items():
                metrics=common.metrics(cell.y[K:],pred)
                path=cell_dir/f'{setting}.npz'
                np.savez_compressed(path,y=cell.y[K:],pred=pred,cycle=cell.cycle[K:])
                artifacts.append(path)
                rows.append(dict(dataset=dataset,domain=domain,cell_id=cell.id,model=name,seed=seed,
                    setting=setting,selected_mode=chosen['primary']['mode'],**metrics,prediction_file=str(path)))
            fits.append(dict(cell_id=cell.id,**fit_log))
        write_json(dest/'training.json',dict(pretrain=info,chosen=chosen,support_fits=fits,boundary=boundary,
                                            reused_source=old['hashes'] if old else None))
        write_csv(dest/'metrics.csv',rows)
        artifacts.extend([dest/'training.json',dest/'metrics.csv',tuning])
        done=dict(status='COMPLETE',dataset=dataset,domain=domain,model=name,seed=seed,
            cells=len(te),rows=len(rows),elapsed_seconds=time.monotonic()-started,
            protocol_sha256=digest(PROTOCOL_PATH),provenance_sha256=digest(provenance_path),artifacts=sha_files(artifacts))
        write_json(complete,done)
        print('COMPLETE',dataset,domain,name,seed,'cells',len(te),'seconds',round(done['elapsed_seconds'],1),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset');ap.add_argument('--domain');ap.add_argument('--model')
    ap.add_argument('--workers',type=int,default=6);ap.add_argument('--plan-only',action='store_true')
    args=ap.parse_args()
    if args.dataset:
        with threadpool_limits(limits=1):run_fold(args.dataset,args.domain,args.model)
        return
    preservation()
    tasks=[(ds,domain,name) for ds in PROTOCOL['datasets'] for domain in sorted({c.domain for c in data.load_cells(ds)})
           for name in PROTOCOL['baseline_models']]
    OUT.mkdir(exist_ok=True)
    write_json(OUT/'plan.json',dict(tasks=tasks,fold_model_tasks=len(tasks),seed_tasks=len(tasks)*3,
        protocol_sha256=digest(PROTOCOL_PATH),models=PROTOCOL['baseline_models'],seeds=PROTOCOL['baseline_seeds']))
    if args.plan_only:
        print('Planned',len(tasks),'fold-model tasks;',len(tasks)*3,'seed tasks');return
    def launch(task):
        ds,domain,name=task;log=OUT/'logs'/f'{ds}__{domain}__{name}.log';log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with log.open('a') as stream:
            p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--dataset',ds,'--domain',domain,'--model',name],
                               stdout=stream,stderr=subprocess.STDOUT,env=env)
            write_json(OUT/'processes'/f'{ds}__{domain}__{name}.json',dict(pid=p.pid,task=task,started_unix=time.time(),log=str(log)))
            rc=p.wait()
        return dict(task=task,returncode=rc,log=str(log))
    done=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(launch,t) for t in tasks]):
            result=future.result();done.append(result)
            write_json(OUT/'progress.json',dict(completed=done,total=len(tasks),failed=sum(r['returncode']!=0 for r in done)))
            print('TASK',len(done),'/',len(tasks),result,flush=True)
    if any(r['returncode']!=0 for r in done):raise SystemExit('Some tasks failed; evidence and logs retained')


if __name__=='__main__':main()
