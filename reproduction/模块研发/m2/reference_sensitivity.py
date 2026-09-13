"""Fixed-embedding source-risk bootstrap sensitivity, not retraining evidence."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import repair_diagnostics as rd
import reference_gate as rg


def risk_seeds(start=101,count=10):
    if start<0 or count<1: raise ValueError('Seed start must be nonnegative and count positive')
    return list(range(start,start+count))


def reweight(head,byid,seed):
    rng=np.random.default_rng(seed);groups=defaultdict(list)
    for i,cid in enumerate(head['source_ids']): groups[byid[cid].domain].append(i)
    counts=np.zeros(len(head['source_ids']),dtype=int)
    weight=np.zeros(len(counts))
    for indices in groups.values():
        draw=rng.choice(indices,len(indices),replace=True)
        for i in draw: counts[i]+=1
        weight[indices]=counts[indices]/len(indices)/len(groups)
    active=weight>0;h=copy.copy(head)
    h['weight']=weight[active];h['costs']=head['costs'][active]
    if 'state_costs' in head: h['state_costs']=head['state_costs'][active]
    h['source_ids']=[cid for cid,a in zip(head['source_ids'],active,strict=True) if a]
    h['views']={name:dict(v,z=v['z'][active]) for name,v in head['views'].items()}
    if 'bagged_seeds' in head:
        h['source_domains']=[d for d,a in zip(head['source_domains'],active,strict=True) if a]
        h['risk_bootstrap_weights']=rg.bootstrap_weights(h)
    return h,counts.tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--seed-start',type=int,default=101);ap.add_argument('--seed-count',type=int,default=10)
    args=ap.parse_args();seeds=risk_seeds(args.seed_start,args.seed_count)
    candidate=Path(args.candidate).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),out/'reference_sensitivity.py')
    obj=json.loads((candidate/'candidate.json').read_text());req=json.loads((candidate/'request.json').read_text())
    assert req['status']=='COMPLETE';screen=Path(req['screen'])
    rd.sa.write_json(out/'protocol.json',dict(status='RUNNING',seeds=seeds,candidate=str(candidate),
        change='Bootstrap source cells within each source domain; alter risk weights only.',
        fixed='GPRs, source feature scaler/PCA, risk values, coverage cloud/thresholds, source state centers/width when present, and selected bandwidth/risk-mixture settings.',
        limits='Development sensitivity experiment, not new data or full retraining stability.',
        code_sha256=rd.sa.digest(Path(__file__))))
    cells=[c for ds in rd.sa.PROTOCOL['datasets'] for c in rd.sa.load_cells(ds)];byid={c.id:c for c in cells}
    rows=[];resamples=[]
    with threadpool_limits(limits=1):
        for item in obj['manifest']:
            fold=item['fold'];group=item['group'];modelpath=candidate/item['artifact']
            assert rd.sa.digest(modelpath)==item['sha256'];model=joblib.load(modelpath)
            _,_,test=rd.sa.split(cells,fold)
            # Stable fold-dependent generator shared by the two parent groups.
            foldcode=sorted({x['fold'] for x in obj['manifest']}).index(fold)*10000
            heads={}
            for seed in seeds:
                h,counts=reweight(model.head,byid,seed+foldcode)
                heads[seed]=h
                resamples.append(dict(fold=fold,group=group,seed=seed,source_ids=model.head['source_ids'],counts=counts))
            view=model.selection['reference_view'];index=model.selection['reference_index']
            input_view=view.removesuffix('_consensus').removesuffix('_supported').removesuffix('_bagged').removesuffix('_soft')
            for c in test:
                path=screen/'folds'/fold.replace(':','__')/'seed_0'/group/(c.id+'.npz')
                with np.load(path) as z:
                    base=z['parent'];physical=z['physical_control'];y=z['y']
                _,res=rd.sa.predict_components(model.parent,rd.sa.prefix(c,rd.sa.K),model.parent_mode)
                extra={}
                if model.head.get('raw_reference',False):
                    rawpath=screen/'folds'/fold.replace(':','__')/'seed_0'/'Base+M2'/(c.id+'.npz')
                    with np.load(rawpath) as z: raw=z['parent']
                    _,rawres=rd.sa.predict_components(model.raw_model,rd.sa.prefix(c,rd.sa.K),model.raw_mode)
                    extra=dict(raw_base=raw,raw_residual=rawres)
                for seed,h in heads.items():
                    slim=dict(h,views={input_view:h['views'][input_view]})
                    e=rg.attach(dict(base=base,physical=physical,residual=res,**extra),c,slim)
                    p=e['reference_predictions'][view][index]
                    rows.append(dict(seed=seed,fold=fold,dataset=c.dataset,domain=c.domain,cell_id=c.id,group=group,**rd.sa.metrics(y,p)))
            print(fold,group,'risk bootstraps scored',flush=True)
    summary=[];gates=[];controls={(r['dataset'],r['group']):r for r in obj['table']}
    for seed in seeds:
        current={}
        for ds in rd.sa.PROTOCOL['datasets']:
            for g in ['Base+M2','Base+M1+M2']:
                rr=[r for r in rows if r['seed']==seed and r['dataset']==ds and r['group']==g]
                assert len(rr)==rd.sa.PROTOCOL['expected_cells'][ds]
                r=dict(seed=seed,dataset=ds,group=g,**{k:rd.sa.macro(rr,k) for k in rd.KEYS})
                summary.append(r);current[ds,g]=r
        gates.append(dict(seed=seed,
            independent=all(current[d,'Base+M2'][k]<controls[d,'Base'][k] for d in rd.sa.PROTOCOL['datasets'] for k in rd.KEYS),
            incremental=all(current[d,'Base+M1+M2'][k]<controls[d,'Base+M1'][k] for d in rd.sa.PROTOCOL['datasets'] for k in rd.KEYS),
            complementary=all(current[d,'Base+M1+M2'][k]<current[d,'Base+M2'][k] for d in rd.sa.PROTOCOL['datasets'] for k in rd.KEYS)))
    rd.sa.write_json(out/'summary.json',dict(status='COMPLETE',summary=summary,gates=gates,rows=rows,resamples=resamples))
    p=json.loads((out/'protocol.json').read_text());p['status']='COMPLETE';rd.sa.write_json(out/'protocol.json',p)
    print(json.dumps(gates),flush=True)


if __name__=='__main__': main()
