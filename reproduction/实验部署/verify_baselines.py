"""Independent saved-checkpoint replay, training/selection and split audit.

Does not invoke the study runner's infer/fresh_source functions. Source models
are never trained. A deterministic support-fit replay uses only ten labels.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np
from study import HERE,ROOT,BUNDLE,PROTOCOL,PROTOCOL_PATH,digest,write_json,write_csv,preservation

sys.path.insert(0,str(ROOT/'模块研发/m1'))
import data
sys.path.insert(0,str(ROOT/'开发基线选择依据/fair_selection'))
import neural
import torch

OUT=HERE/'baselines'


def replay(task):
    torch.set_num_threads(1)
    ds,domain,name=task;job=OUT/f'{ds}__{domain}__{name}'
    prov=json.loads((job/'provenance.json').read_text());chosen=json.loads((job/'chosen.json').read_text())
    cells=data.load_cells(ds);tr,va,te=data.split(cells,ds+':'+domain)
    for k,group in [('train_cells',tr),('validation_cells',va),('target_cells',te)]:assert prov[k]==[c.id for c in group]
    idx=data.source_indices(tr);tr=sorted(tr,key=lambda c:data.natural(c.id))
    source=np.concatenate([c.x[idx[c.id]] for c in tr]);mu=source.mean((0,2),keepdims=True);sd=source.std((0,2),keepdims=True);sd[sd<1e-6]=1
    np.testing.assert_array_equal(mu,np.asarray(prov['normalization']['mean'],dtype='float32'))
    np.testing.assert_array_equal(sd,np.asarray(prov['normalization']['std'],dtype='float32'))
    assert {(c.id,int(j)) for c in tr for j in idx[c.id]}=={tuple(k) for k in prov['source_positional_keys']}
    assert len(source)<=1000 and len(tr)<=60
    norm=lambda x:((x-mu)/sd).astype('float32')
    architecture='MLP' if name=='MAML_MLP' else name
    if name!='MAML_MLP':assert chosen['primary']==min(chosen['mode_scores'],key=lambda r:r['inner_mae'])
    else:assert chosen['primary']['steps']==min(chosen['trials'],key=lambda r:r['inner_mae'])['steps']
    results=[];curves={};validation_rows=[]
    for seed in PROTOCOL['baseline_seeds']:
        dest=job/f'seed_{seed}';record=json.loads((dest/'training.json').read_text());info=record['pretrain']
        assert info['stop_reason']=='validation_patience'
        if name=='MAML_MLP':
            assert 500<=info['outer_updates']<=4000 and info['outer_updates']-info['best_step']>=400
        else:
            assert 100<=info['epochs_run']<=1200 and info['epochs_run']-info['best_epoch']>=40
            path=Path(info['learning_curve']);curve=list(csv.DictReader(path.open()));curves[str(path)]=digest(path)
            best=min(curve,key=lambda r:float(r['validation_macro_mae']))
            assert int(best['epoch'])==info['best_epoch'] and int(curve[-1]['epoch'])==info['epochs_run']
            assert abs(float(best['validation_macro_mae'])-info['best_inner_mae'])<1e-12
        base=neural.MODELS[architecture]();base.load_state_dict(torch.load(dest/'source.pt',map_location='cpu',weights_only=True));base.eval()
        validation=[]
        for c in va:
            if name=='MAML_MLP':
                fitted,_=neural.fit_support(base,'MLP',norm(c.x[:10]),c.y[:10],'full_finetune',.0005,50,seed+10000,'cpu')
                p=neural.predict(fitted,norm(c.x[10:]))
            else:p=neural.predict(base,norm(c.x[10:]))
            validation.append(float(abs(p-c.y[10:]).mean()))
        delta=abs(float(np.mean(validation))-info['best_inner_mae']);assert delta<1e-6,(task,seed,'validation',delta)
        validation_rows.append(dict(dataset=ds,domain=domain,model=name,seed=seed,replayed_validation_mae=float(np.mean(validation)),stored_validation_mae=info['best_inner_mae'],absolute_difference=delta))
        for n,c in enumerate(te):
            folder=dest/'cells'/c.id;mode=chosen['primary']['mode'];a=norm(c.x)
            source_pred=neural.predict(base,a)
            fitted=neural.MODELS[architecture]();state=torch.load(folder/'adapted.pt',map_location='cpu',weights_only=True);fitted.load_state_dict(state);fitted.eval()
            primary=neural.predict(fitted,a[10:])
            bias=float((c.y[:10]-source_pred[:10]).mean())
            if mode=='source_bias':primary+=bias
            expected={'inner_selected':primary,'source_only':source_pred[10:],'source_bias':source_pred[10:]+bias}
            differences={}
            for setting,p in expected.items():
                with np.load(folder/(setting+'.npz'),allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['y'],c.y[10:]);np.testing.assert_array_equal(z['cycle'],c.cycle[10:])
                    difference=float(np.max(abs(p-z['pred'])))
                assert difference<1e-5,(task,seed,c.id,setting,difference)
                differences[setting]=difference
            support_state_error=0.
            if n==0 and mode not in ['source_only','source_bias']:
                rebuilt,_=neural.fit_support(base,architecture,a[:10],c.y[:10],mode,chosen['primary']['lr'],chosen['primary']['steps'],seed+10000,'cpu')
                support_state_error=max(float((state[k]-v).abs().max()) for k,v in rebuilt.state_dict().items())
                assert support_state_error==0,(task,seed,'support_state',support_state_error)
            individual=neural.predict(fitted,a[[10,-1]])
            if mode=='source_bias':individual+=bias
            current_error=float(np.max(abs(individual-primary[[0,-1]])))
            assert current_error<1e-5
            results.append(dict(dataset=ds,domain=domain,model=name,seed=seed,cell_id=c.id,selected_mode=mode,
                query_points=len(c.y)-10,prediction_replay_max_abs=max(differences.values()),
                current_query_max_abs=current_error,support_fit_replayed=n==0 and mode not in ['source_only','source_bias'],
                support_state_max_abs=support_state_error))
    result=dict(task=task,status='VERIFIED',cell_seed_rows=results,validation_rows=validation_rows,learning_curve_sha256=curves,
        source_labels=len(source),source_cells=len(tr),source_validation_cells=len(va),no_source_training_performed=True,
        provenance_sha256=digest(job/'provenance.json'),chosen_sha256=digest(job/'chosen.json'))
    write_json(OUT/'replay_audit'/('___'.join(task)+'.json'),result)
    return result


def main():
    preservation();plan=json.loads((OUT/'plan.json').read_text());hashes={}
    manifest=json.loads((BUNDLE/'candidate.json').read_text())
    for ds,domain,name in plan['tasks']:
        prov=json.loads((OUT/f'{ds}__{domain}__{name}'/'provenance.json').read_text())
        assert prov['protocol_sha256']==digest(PROTOCOL_PATH)
        reference=next(r for r in manifest['manifest'] if r['fold']==ds+':'+domain and r['group']=='B1234')
        assert {tuple(k) for k in reference['source_keys']}=={tuple(k) for k in prov['source_positional_keys']}
        for p,h in {**prov['data_sha256'],**prov['code_sha256']}.items():
            if p in hashes:assert hashes[p]==h
            else:assert digest(p)==h,p;hashes[p]=h
    done=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(replay,t) for t in plan['tasks']]):
            r=f.result();done.append(r);print('REPLAY',len(done),'/',len(plan['tasks']),r['task'],flush=True)
    rows=[r for v in done for r in v['cell_seed_rows']];validation=[r for v in done for r in v['validation_rows']]
    dest=OUT/'summary';write_csv(dest/'checkpoint_replay.csv',rows);write_csv(dest/'validation_replay.csv',validation)
    report=dict(status='ALL_CHECKPOINTS_AND_SOURCE_VALIDATION_VERIFIED',fold_model_tasks=len(done),seed_tasks=len(validation),
        primary_cell_seed_predictions=len(rows),all_setting_predictions=len(rows)*3,
        source_training_performed=False,all_source_universes_match_frozen_final=True,
        training_stop_reasons=dict(Counter('validation_patience' for _ in validation)),
        max_prediction_replay_abs=max(r['prediction_replay_max_abs'] for r in rows),
        max_current_query_abs=max(r['current_query_max_abs'] for r in rows),
        support_fit_replays=sum(r['support_fit_replayed'] for r in rows),
        max_validation_replay_abs=max(r['absolute_difference'] for r in validation),
        input_hashes=hashes,code_sha256=digest(Path(__file__)),
        tolerance='1e-5 SOH ratio for float32 batch-shape numerical differences; source validation1e-6',
        limits='Replay validates recorded local implementations and declared budgets, not a claim of globally optimal tuning or official benchmark reproduction')
    write_json(dest/'checkpoint_replay_audit.json',report);print('ALL_REPLAYS_VERIFIED',report['seed_tasks'],report['primary_cell_seed_predictions'],flush=True)


if __name__=='__main__':main()
