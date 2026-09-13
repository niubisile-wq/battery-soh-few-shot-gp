"""Shared validation-only source-coverage fusion with same-weight ungated controls."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,SCREEN,GROUPS,aggregate,error_metrics,dump_csv
from coverage_gate import coverage,trust
from summarize_strict import paired_stats

P=json.loads((HERE/'coverage_protocol.json').read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((FROZEN/'candidate.json').read_text())
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    sa.write_json(out/'run_protocol.json',dict(protocol=P,hashes={p.name:sa.digest(p) for p in [HERE/'coverage_protocol.json',HERE/'coverage_gate.py',Path(__file__)]}))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)]
    rows=[];selections=[];boundary=0.;hashes={};gate_stats=[]
    with threadpool_limits(limits=1):
        for fold in sorted({r['fold'] for r in manifest['manifest']}):
            tr,val,test=sa.split(cells,fold);name=fold.replace(':','__');dest=out/'folds'/name
            entry=next(r for r in manifest['manifest'] if r['fold']==fold and r['group']=='Base+M1+M2')
            job=SCREEN/name/'seed_0';ee=joblib.load(job/'Base+M1+M2_selection_episodes.joblib')
            sa.check_validation(ee,[c.id for c in val],[c.id for c in test]);byid={e['cell_id']:e for e in ee};s=entry['selection']
            vp={c.id:byid[c.id]['reference_predictions'][s['reference_view']][s['reference_index']] for c in val}
            scores=[];trials=[]
            for source,views in P['sources'].items():
                src=HERE.parent/'results'/source/'folds'/name
                for path in sorted(src.glob('*.joblib')):
                    if not any(path.name.startswith(v+'__') for v in views):continue
                    model=joblib.load(path)
                    assert set(model.source_keys)=={tuple(k) for k in entry['source_keys']}
                    pred={c.id:model.predict(sa.inference_view(c)) for c in val}
                    cov={c.id:coverage(model,sa.inference_view(c)) for c in val}
                    for kind,power in P['gates']:
                        for w in P['weights']:
                            loss=sa.macro([dict(dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(
                                vp[c.id]+w*trust(*cov[c.id],kind,power)*(pred[c.id]-vp[c.id])-c.y[sa.K:])))) for c in val])
                            scores.append((loss,w,str(path),kind,power))
                            trials.append(dict(path=str(path),kind=kind,power=power,weight=w,validation_mae=loss))
            loss,w,path,kind,power=min(scores);chosen=dict(path=path,kind=kind,power=power,weight=w,validation_mae=loss)
            sa.write_json(dest/'selection.json',dict(chosen=chosen,trials=trials,validation_cells=[c.id for c in val],source_keys=entry['source_keys']))
            selections.append(dict(fold=fold,chosen=chosen));hashes[path]=sa.digest(path);model=joblib.load(path)
            for c in test:
                p=model.predict(sa.inference_view(c));q,r=coverage(model,sa.inference_view(c));t=trust(q,r,kind,power)
                yy=c.y.copy();yy[sa.K:]=123;n=min(len(c.x),sa.K+3)
                altered=replace(c,y=yy);p2=model.predict(altered);q2,r2=coverage(model,altered)
                short=sa.prefix(c,n);ps=model.predict(short);qs,rs=coverage(model,short)
                boundary=max(boundary,float(np.max(abs(p-p2))),float(np.max(abs(t-trust(q2,r2,kind,power)))),
                    float(np.max(abs(p[:n-sa.K]-ps))),float(np.max(abs(t[:n-sa.K]-trust(qs,rs,kind,power)))))
                assert boundary<1e-8
                parent={}
                for g,raw in [('Base+M2','B'),('Base+M1+M2','B1')]:
                    with np.load(job/g/(c.id+'.npz')) as z:
                        assert np.array_equal(z['y'],c.y[sa.K:]);parent[raw]=z['parent'].copy()
                        parent['B2' if raw=='B' else 'B12']=z[manifest['variant']].copy()
                pred=dict(parent)
                for g in GROUPS:
                    pred[g+'3']=parent[g]+w*t*(p-parent[g])
                    pred[g+'3_ungated']=parent[g]+w*(p-parent[g])
                gate_stats.append(dict(cell_id=c.id,dataset=c.dataset,domain=c.domain,reference_coverage=r,mean_query_coverage=float(np.mean(q)),mean_weight=float(np.mean(w*t))))
                fp=dest/'predictions'/(c.id+'.npz');fp.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(fp,y=c.y[sa.K:],branch=p,trust=t,**pred)
                for g,pp in pred.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(c.y[sa.K:],pp)))
            print(fold,'coverage screen complete',flush=True)
    assert len(rows)==4380
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    edges=[('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]
    gates={a+'<'+b:all(look[ds,a][m]<look[ds,b][m] for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae']) for a,b in edges}
    reverse={v:k for k,v in GROUPS.items()};err=max(abs(look[r['dataset'],reverse[r['group']]][m]-100*r[m]) for r in manifest['table'] for m in ['mae','rmse','p95_ae'])
    assert err<1e-8 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    assert all(sa.digest(path)==h for path,h in hashes.items())
    pairs=[]
    for a,b in edges+[('B123','B123_ungated')]:
        for ds in ['XJTU','MATR','Tongji']:
            aa={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==a};bb={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==b}
            for m in ['mae','rmse','p95_ae']:pairs.append(dict(comparison=a+'-'+b,dataset=ds,metric=m,
                **paired_stats([(aa[c][m]-bb[c][m])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table);dump_csv(out/'coverage.csv',gate_stats)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,selections=selections,paired_comparisons=pairs,
        max_boundary_error=boundary,max_frozen_parent_metric_error=err,source_hashes=hashes,cells=365,folds=21))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
