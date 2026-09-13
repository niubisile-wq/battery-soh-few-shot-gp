"""Decompose previous risk perturbation losses and test binary bagged fallback."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import nested_shrink_sensitivity as ss


def binary_weight(costs,domains,params):
    parent=next(j for j,p in enumerate(params) if p['beta']==0 and p['physical']==0 and p.get('raw_mix',0)==0)
    physical=next(j for j,p in enumerate(params) if p['beta']==0 and p['physical']==1 and p.get('raw_mix',0)==0)
    domains=np.asarray(domains);wins=[]
    for seed in range(256):
        ix=ss.sample_rows(domains,seed)
        risk=np.mean([costs[ix][domains[ix]==d][:,[parent,physical]].mean(0) for d in sorted(set(domains))],axis=0)
        wins.append(int(np.lexsort((risk[:,2],risk[:,1],risk[:,0]))[0]))
    return float(np.mean(wins))


def decomposition(summary):
    grouped=defaultdict(dict)
    for r in summary:grouped[r['seed'],r['parent']][r['method']]=r
    result=[]
    for (seed,parent),g in sorted(grouped.items()):
        for k in ss.sp.METRICS:
            switch=g['simple'][k]-g['fixed_simple'][k]
            correction=g['domain_guarded'][k]-g['simple'][k]
            total=g['domain_guarded'][k]-g['fixed_simple'][k]
            np.testing.assert_allclose(switch+correction,total,rtol=0,atol=1e-12)
            result.append(dict(seed=seed,parent=parent,metric=k,switch=switch,correction=correction,total=total))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=ss.sp.ne.ns.rd.sa
    root=sa.ROOT/'模块研发/results';oldroot=root/'m2_nested_shrink_sensitivity_v1'
    audit=json.loads((oldroot/'audit.json').read_text());assert audit['status']=='PASS'
    assert audit['result_sha256']==sa.digest(oldroot/'result.json')
    old=json.loads((oldroot/'result.json').read_text());parts=decomposition(old['summary'])
    source=json.loads((root/'m2_nested_shrink_probe_v2/result.json').read_text())
    rows=[];weights=[];seeds=[0]+list(range(2001,2011))
    for c in source['choices']:
        folder=root/'m2_nested_signal_probe_v1/folds'/c['fold'].replace(':','__')
        original=joblib.load(folder/'source_lodo_episodes.joblib')
        validation=[e for e in ss.sp.ne.rg.with_raw_reference(original[c['parent']],original['Base']) if e['domain']==c['held']]
        assert [e['cell_id'] for e in validation]==c['validation_ids']
        for seed in seeds:
            ix=np.arange(len(c['domains'])) if seed==0 else ss.sample_rows(c['domains'],seed)
            w=binary_weight(np.array(c['costs'])[ix],np.array(c['domains'])[ix],c['params'])
            weights.append(dict(fold=c['fold'],held=c['held'],parent=c['parent'],seed=seed,physical_weight=w))
            for e in validation:
                prediction=(1-w)*e['base']+w*e['physical'];m=sa.metrics(e['y'],prediction)
                rows.append(dict(seed=seed,fold=c['fold'],domain=c['held'],parent=c['parent'],method='binary_bagged',cell_id=e['cell_id'],
                                 **{k:float(100*m[k]) for k in ss.sp.METRICS}))
    summary=ss.summarize(rows);control={(r['seed'],r['parent'],r['method']):r for r in old['summary']}
    unperturbed={(r['parent'],r['method']):r for r in source['summary']};gates=[]
    for seed in seeds:
        for method in ['simple','fixed_simple']:
            diffs={}
            for parent in ['Base','Base+M1']:
                r=next(r for r in summary if (r['seed'],r['parent'])==(seed,parent))
                b=unperturbed[parent,'simple'] if seed==0 else control[seed,parent,method]
                diffs[parent]={k:r[k]-b[k] for k in ss.sp.METRICS}
            gates.append(dict(seed=seed,control=method,differences=diffs,strict=all(v<0 for d in diffs.values() for v in d.values())))
    sa.write_json(out/'result.json',dict(status='COMPLETE',decomposition=parts,summary=summary,rows=rows,weights=weights,gates=gates,
        input_hashes={str(p):sa.digest(p) for p in [oldroot/'result.json',root/'m2_nested_shrink_probe_v2/result.json']},
        limits=['Source-only repeated MATR development, fixed GP fits. Seed 0 means unperturbed risk table.',
                'Metric decomposition is an exact telescoping identity, not a causal attribution.',
                'Binary vote average is a new simple routing pilot, not v8 or a repaired full M2.']))
    for p in [Path(__file__),Path(ss.__file__)]:shutil.copyfile(p,out/p.name)
    for parent in ['Base','Base+M1']:
        print(parent,'MAE decomposition', {k:float(np.mean([r[k] for r in parts if r['parent']==parent and r['metric']=='mae'])) for k in ['switch','correction','total']})
    for method in ['simple','fixed_simple']:
        print(method,'strict',sum(g['strict'] for g in gates if g['seed'] and g['control']==method),'/10')
    print('unperturbed',json.dumps([r for r in summary if r['seed']==0]))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
