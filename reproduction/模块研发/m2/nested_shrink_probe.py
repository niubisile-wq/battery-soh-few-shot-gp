"""Nested source pilot: shrink uncertain expert corrections toward a simple selector."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import nested_expert_probe as ne

METRICS=['mae','rmse','p95_ae']


def choose(costs,domains,params,n_bags=256):
    domains=np.asarray(domains);unique=sorted(set(domains))
    weights=np.array([1/len(unique)/np.sum(domains==d) for d in domains])
    risk=np.einsum('i,ijk->jk',weights,costs)
    simple=[j for j,p in enumerate(params) if p['beta']==0 and p.get('raw_mix',0)==0 and p['physical'] in [0.,1.]]
    reference=min(simple,key=lambda j:tuple(risk[j]))
    selected=int(ne.rg.risk_indices(risk,params,'mae'))
    draws=[]
    for seed in range(n_bags):
        rng=np.random.default_rng(seed);w=np.zeros(len(domains))
        for d in unique:
            idx=np.flatnonzero(domains==d);chosen=rng.choice(idx,len(idx),replace=True)
            w+=np.bincount(chosen,minlength=len(w))/len(idx)/len(unique)
        draws.append(np.einsum('i,ijk->jk',w,costs))
    draws=np.array(draws);gain=draws[:,reference]-draws[:,selected]
    mean_gain=risk[reference]-risk[selected];lower=np.quantile(gain,.1,axis=0)
    # A heuristic stability coefficient, not a confidence interval guarantee.
    fractions=np.divide(np.maximum(lower,0),np.maximum(mean_gain,1e-12))
    coefficient=float(np.clip(fractions.min(),0,1))
    if selected==reference: coefficient=0.
    domain_gain=np.array([costs[domains==d,reference].mean(0)-costs[domains==d,selected].mean(0) for d in unique])
    domain_coefficient=float(min(coefficient,np.clip(np.min(np.maximum(domain_gain.min(0),0)/np.maximum(mean_gain,1e-12)),0,1)))
    bag_indices=ne.rg.risk_indices(draws,params,'mae')
    return dict(reference=int(reference),selected=selected,coefficient=coefficient,domain_coefficient=domain_coefficient,
                domain_gains=domain_gain.tolist(),
                mean_gain=mean_gain.tolist(),lower_gain=lower.tolist(),bag_indices=bag_indices.tolist())


def predictions(e,params,choice):
    simple=ne.rg.expert_prediction(e,params[choice['reference']])
    complex_=ne.rg.expert_prediction(e,params[choice['selected']])
    bagged=np.mean([ne.rg.expert_prediction(e,params[j]) for j in choice['bag_indices']],axis=0)
    return dict(simple=simple,complex=complex_,bagged=bagged,
                guarded=simple+choice['coefficient']*(complex_-simple),
                domain_guarded=simple+choice.get('domain_coefficient',choice['coefficient'])*(complex_-simple))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=ne.ns.rd.sa
    root=sa.ROOT/'模块研发/results/m2_nested_signal_probe_v1'
    assert json.loads((root/'audit.json').read_text())['status']=='PASS'
    protocol=json.loads((root/'protocol.json').read_text());cells=sa.load_cells('MATR');rows=[];choices=[]
    for fold in protocol['folds']:
        train,_,_=sa.split(cells,fold);byid={c.id:c for c in train};folder=root/'folds'/fold.replace(':','__')
        result=json.loads((folder/'result.json').read_text());pairs={}
        for excluded in sorted({tuple(f['excluded']) for f in result['fits']}):
            fits=[f for f in result['fits'] if tuple(f['excluded'])==excluded];models={}
            for f in fits:
                if f['group']=='Physical_control': continue
                path=root/f['artifact'];assert sa.digest(path)==f['sha256'];models[f['group']]=joblib.load(path)
            episodes=joblib.load((root/fits[0]['artifact']).parent/'episodes.joblib')
            pairs[excluded]={p:ne.enrich(episodes[p],episodes['Base'],models,byid,p) for p in episodes}
        original=joblib.load(folder/'source_lodo_episodes.joblib')
        for held in sorted({c.domain for c in train}):
            for parent in ['Base','Base+M1']:
                training=[]
                for domain in sorted({c.domain for c in train}-{held}):
                    training.extend(e for e in pairs[tuple(sorted([held,domain]))][parent] if e['domain']==domain)
                validation=[e for e in ne.rg.with_raw_reference(original[parent],original['Base']) if e['domain']==held]
                assert not {e['cell_id'] for e in training}&{e['cell_id'] for e in validation}
                params=[dict(p,raw_mix=m) for m in ([0.] if parent=='Base' else [0.,.5,1.]) for p in ne.rg.PARAMS]
                costs=np.array([[ne.rg.expert_risk(e,p) for p in params] for e in training])
                choice=choose(costs,[e['domain'] for e in training],params)
                choices.append(dict(fold=fold,held=held,parent=parent,params=params,choice=choice,
                    train_ids=[e['cell_id'] for e in training],validation_ids=[e['cell_id'] for e in validation],
                    domains=[e['domain'] for e in training],costs=costs.tolist()))
                for e in validation:
                    for method,p in predictions(e,params,choice).items():
                        m=sa.metrics(e['y'],p)
                        rows.append(dict(fold=fold,domain=held,parent=parent,method=method,cell_id=e['cell_id'],
                                         **{k:float(m[k]*100) for k in METRICS}))
    summary=[]
    for parent in ['Base','Base+M1']:
        for method in ['simple','complex','bagged','guarded','domain_guarded']:
            groups=defaultdict(list)
            for r in rows:
                if r['parent']==parent and r['method']==method:groups[r['fold'],r['domain']].append(r)
            summary.append(dict(parent=parent,method=method,**{k:float(np.mean([np.mean([r[k] for r in g]) for g in groups.values()])) for k in METRICS}))
    sa.write_json(out/'result.json',dict(status='COMPLETE',summary=summary,rows=rows,choices=choices,
        parameters=dict(bootstrap=256,lower_quantile=.1),limits=['MATR nested source fitting only; inherited outer configurations; no outer test scoring.',
        'Risk bootstrap fixes fitted GPs; not independent refitting or new data.',
        'Quantile shrinkage is a heuristic, not selection-adjusted confidence coverage or guaranteed improvement.',
        'This global routing pilot omits v8 reference conditioning and coverage gate.']))
    for p in [Path(__file__),Path(ne.__file__)]: shutil.copyfile(p,out/p.name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
