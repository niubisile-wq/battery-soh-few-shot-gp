"""Fixed-GP, paired source-cell risk sensitivity for nested shrink pilot."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import nested_shrink_probe as sp


def sample_rows(domains,seed):
    domains=np.asarray(domains);rng=np.random.default_rng(seed);indices=[]
    for domain in sorted(set(domains)):
        idx=np.flatnonzero(domains==domain)
        indices.extend(rng.choice(idx,len(idx),replace=True).tolist())
    return np.array(indices,dtype=int)


def summarize(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['seed'],r['parent'],r['method'],r['fold'],r['domain']].append(r)
    pooled=defaultdict(list)
    for (seed,parent,method,fold,domain),rr in grouped.items():
        pooled[seed,parent,method].append({k:float(np.mean([r[k] for r in rr])) for k in sp.METRICS})
    return [dict(seed=seed,parent=parent,method=method,**{k:float(np.mean([r[k] for r in rr])) for k in sp.METRICS})
            for (seed,parent,method),rr in sorted(pooled.items())]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=sp.ne.ns.rd.sa
    root=sa.ROOT/'模块研发/results/m2_nested_shrink_probe_v2'
    assert json.loads((root/'audit.json').read_text())['status']=='PASS'
    assert sa.digest(root/'nested_shrink_probe.py')==sa.digest(Path(sp.__file__))
    source=json.loads((root/'result.json').read_text());seeds=list(range(2001,2011))
    protocol=dict(status='RUNNING',seeds=seeds,input_sha256=sa.digest(root/'result.json'),
        limits=['MATR source pilot only; no outer target scores or full GP retraining.',
                'Both parents use identical paired source-cell resampling; held source scoring cells remain fixed.',
                'Compare both resampled and unchanged simple controls to avoid credit from weakening controls.',
                'Fixed domain consistency and q10 shrinkage; repeated development data, not independent confirmation.'])
    sa.write_json(out/'protocol.json',protocol)
    rows=[];draws=[];paired={}
    for record in source['choices']:
        folder=sa.ROOT/'模块研发/results/m2_nested_signal_probe_v1/folds'/record['fold'].replace(':','__')
        original=joblib.load(folder/'source_lodo_episodes.joblib')
        validation=[e for e in sp.ne.rg.with_raw_reference(original[record['parent']],original['Base']) if e['domain']==record['held']]
        assert [e['cell_id'] for e in validation]==record['validation_ids']
        assert not set(record['train_ids'])&set(record['validation_ids'])
        for seed in seeds:
            indices=sample_rows(record['domains'],seed);ids=[record['train_ids'][j] for j in indices]
            key=(record['fold'],record['held'],seed)
            if key in paired:assert paired[key]==ids
            else:paired[key]=ids
            choice=sp.choose(np.array(record['costs'])[indices],np.array(record['domains'])[indices],record['params'])
            draws.append(dict(fold=record['fold'],held=record['held'],parent=record['parent'],seed=seed,
                              sample_ids=ids,choice=choice))
            for e in validation:
                simple=sp.ne.rg.expert_prediction(e,record['params'][choice['reference']])
                complex_=sp.ne.rg.expert_prediction(e,record['params'][choice['selected']])
                fixed=sp.ne.rg.expert_prediction(e,record['params'][record['choice']['reference']])
                for method,p in dict(simple=simple,fixed_simple=fixed,domain_guarded=simple+choice['domain_coefficient']*(complex_-simple)).items():
                    metrics=sa.metrics(e['y'],p)
                    rows.append(dict(seed=seed,fold=record['fold'],domain=record['held'],parent=record['parent'],
                                     method=method,cell_id=e['cell_id'],**{k:float(100*metrics[k]) for k in sp.METRICS}))
        print(record['fold'],record['held'],record['parent'],'complete',flush=True)
    summary=summarize(rows);lookup={(r['seed'],r['parent'],r['method']):r for r in summary};gates=[]
    for seed in seeds:
        for control in ['simple','fixed_simple']:
            differences={parent:{k:lookup[seed,parent,'domain_guarded'][k]-lookup[seed,parent,control][k] for k in sp.METRICS} for parent in ['Base','Base+M1']}
            gates.append(dict(seed=seed,control=control,strict_improvement=all(v<0 for d in differences.values() for v in d.values()),
                              no_regression=all(v<=1e-10 for d in differences.values() for v in d.values()),differences=differences))
    sa.write_json(out/'result.json',dict(status='COMPLETE',summary=summary,rows=rows,draws=draws,gates=gates))
    protocol['status']='COMPLETE';sa.write_json(out/'protocol.json',protocol)
    for p in [Path(__file__),Path(sp.__file__)]:shutil.copyfile(p,out/p.name)
    for control in ['simple','fixed_simple']:
        g=[r for r in gates if r['control']==control]
        print(control,'strict',sum(r['strict_improvement'] for r in g),'no regression',sum(r['no_regression'] for r in g),'/',len(g))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
