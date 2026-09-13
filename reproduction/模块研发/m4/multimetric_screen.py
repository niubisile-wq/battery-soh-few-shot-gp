"""Two global validation policies across cached residual families; no new fitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def choose_fold(fold,folders,out,protocol):
    cells=sa.load_cells(fold.split(':')[0]);_,val,test=sa.split(cells,fold)
    vp=joblib.load(folders['plain']/'validation_parents.joblib');other=joblib.load(folders['anchored']/'validation_parents.joblib')
    assert set(vp)==set(other)=={(c.id,g) for c in val for g in PARENTS}
    assert all(np.array_equal(vp[k],other[k]) for k in vp)
    baseline={g:{k:sa.macro([dict(dataset=c.dataset,domain=c.domain,**sa.metrics(c.y[sa.K:],vp[c.id,g])) for c in val],k)
                 for k in protocol['metrics']} for g in protocol['endpoints']}
    trials=[]
    for family,folder in folders.items():
        old=json.loads((folder/'result.json').read_text());keys=sorted({r['key'] for r in old['trials']})
        for key in keys:
            models={g:joblib.load(folder/'models'/(g+'__'+key+'.joblib')) for g in protocol['endpoints']}
            residual={(c.id,g):m.predict(sa.inference_view(c),vp[c.id,g]) for g,m in models.items() for c in val}
            for gain in protocol['gains']:
                mm={g:{k:sa.macro([dict(dataset=c.dataset,domain=c.domain,**sa.metrics(c.y[sa.K:],vp[c.id,g]+gain*residual[c.id,g])) for c in val],k)
                       for k in protocol['metrics']} for g in protocol['endpoints']}
                score=float(np.mean([mm[g][k]/max(baseline[g][k],1e-10) for g in mm for k in protocol['metrics']]))
                feasible=all(mm[g][k]<=baseline[g][k]+1e-12 for g in mm for k in protocol['metrics'])
                trials.append(dict(family=family,key=key,gain=gain,score=score,feasible=feasible,metrics=mm))
    rank=lambda r:(r['score'],r['gain'],r['family'],r['key'])
    choices={'balanced':min(trials,key=rank),'constrained':min([r for r in trials if r['feasible']],key=rank)}
    selected={}
    for policy,choice in choices.items():
        folder=folders[choice['family']];paths={g:folder/'models'/(g+'__'+choice['key']+'.joblib') for g in PARENTS}
        selected[policy]=dict(choice=choice,models={g:dict(path=str(p),sha256=sa.digest(p)) for g,p in paths.items()})
    record=dict(fold=fold,baseline=baseline,trials=trials,selected=selected,validation_ids=[c.id for c in val])
    sa.write_json(out/'folds'/fold.replace(':','__')/'selection.json',record)
    print(fold,'both policies selected on validation only',flush=True);return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);root=HERE.parent/'results'
    p=json.loads((HERE/'multimetric_protocol.json').read_text());folders={}
    for family,dirname in [('plain','m4_source_batch_v1'),('anchored','m4_anchored_batch_v1')]:
        b=json.loads((root/dirname/'result.json').read_text());assert b['status']=='VALIDATION_BATCH_COMPLETE'
        for r in b['folds']:folders.setdefault(r['fold'],{})[family]=Path(r['correction'])
    assert len(folders)==21
    sa.write_json(out/'protocol.json',dict(protocol=p,code_sha256=sa.digest(Path(__file__))))
    selections=[]
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(choose_fold,fold,ff,out,p) for fold,ff in folders.items()]):selections.append(f.result())
    assert len(selections)==21
    sa.write_json(out/'frozen_selections.json',dict(selections=selections,all_choices_saved_before_outer_scoring=True))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];candidate=root/'m3_waveform_candidate_v1'
    frozen=json.loads((candidate/'candidate.json').read_text());rows=[];boundary=0.
    with threadpool_limits(limits=1):
        for selection in selections:
            fold=selection['fold'];name=fold.replace(':','__');_,_,test=sa.split(cells,fold);models={}
            for policy,s in selection['selected'].items():
                models[policy]={g:joblib.load(spec['path']) for g,spec in s['models'].items()}
                assert all(sa.digest(Path(spec['path']))==spec['sha256'] for spec in s['models'].values())
            for c in test:
                with np.load(candidate/'folds'/name/'predictions'/(c.id+'.npz')) as z:
                    y=z['y'];assert np.array_equal(y,c.y[sa.K:]);pp={g:z[g].copy() for g in PARENTS}
                for policy,mm in models.items():
                    gain=selection['selected'][policy]['choice']['gain']
                    for g,m in mm.items():
                        r=m.predict(c,pp[g]);pp[g+'4__'+policy]=pp[g]+gain*r
                        yy=c.y.copy();yy[sa.K:]=123;n=min(len(c.x),sa.K+3)
                        boundary=max(boundary,float(np.max(abs(r-m.predict(replace(c,y=yy),pp[g])))),
                            float(np.max(abs(r[:n-sa.K]-m.predict(sa.prefix(c,n),pp[g][:n-sa.K])))))
                assert boundary<1e-8
                path=out/'folds'/name/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,**pp)
                for g,pred in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,pred)))
            print(fold,'both policies all16 groups scored',flush=True)
    assert len(rows)==8760
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={};failures={}
    for policy in p['policies']:
        tag=lambda g:g+'__'+policy if '4' in g else g
        failures[policy]={a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=look[ds,tag(a)][k]-look[ds,tag(b)][k])
            for ds in ['XJTU','MATR','Tongji'] for k in p['metrics'] if look[ds,tag(a)][k]>=look[ds,tag(b)][k]] for a,b in addition_edges()}
        gates[policy]={k:not v for k,v in failures[policy].items()}
    assert all(sa.digest(candidate/r['artifact'])==r['sha256'] for r in frozen['manifest'])
    assert max(abs(look[r['dataset'],r['group']][k]-r[k]) for r in frozen['table'] for k in p['metrics'])<1e-8
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,failures=failures,max_boundary_error=boundary,
        passing_policies=[k for k,v in gates.items() if all(v.values())],limits=p['limits']))
    print(json.dumps(dict(passing_policies=[k for k,v in gates.items() if all(v.values())])),flush=True)


if __name__=='__main__':main()
