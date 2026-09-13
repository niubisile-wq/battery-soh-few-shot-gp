"""Eight-group validation-selected causal filter screen and direct replay audit."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from screen import sa,HERE,FROZEN,SCREEN,GROUPS,aggregate,error_metrics,dump_csv
from trajectory import correction
from summarize_strict import paired_stats

P=json.loads((HERE/'trajectory_protocol.json').read_text())


def main():
    global P,correction
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    variants=ap.add_mutually_exclusive_group();variants.add_argument('--signal',action='store_true');variants.add_argument('--query-only',action='store_true')
    variants.add_argument('--risk-constrained',action='store_true')
    args=ap.parse_args()
    if args.signal:
        from signal_trajectory import correction as signal_correction
        correction=signal_correction
        P=json.loads((HERE/'signal_trajectory_protocol.json').read_text())
    if args.query_only:
        from query_trajectory import correction as query_correction
        correction=query_correction
        P=json.loads((HERE/'query_trajectory_protocol.json').read_text())
    if args.risk_constrained:
        from query_trajectory import correction as query_correction
        correction=query_correction
        P=json.loads((HERE/'risk_trajectory_protocol.json').read_text())
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((FROZEN/'candidate.json').read_text())
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    paths=[HERE/'trajectory.py',HERE/'trajectory_protocol.json',Path(__file__)]
    if args.signal:paths.extend([HERE/'signal_trajectory.py',HERE/'signal_trajectory_protocol.json'])
    if args.query_only:paths.extend([HERE/'query_trajectory.py',HERE/'query_trajectory_protocol.json'])
    if args.risk_constrained:paths.extend([HERE/'query_trajectory.py',HERE/'risk_trajectory_protocol.json'])
    sa.write_json(out/'run_protocol.json',dict(protocol=P,hashes={p.name:sa.digest(p) for p in paths}))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)]
    rows=[];selections=[];boundary=0.;replay=0.
    for fold in sorted({r['fold'] for r in manifest['manifest']}):
        _,val,test=sa.split(cells,fold);name=fold.replace(':','__');dest=out/'folds'/name
        job=SCREEN/name/'seed_0';entries={r['group']:r for r in manifest['manifest'] if r['fold']==fold}
        cached={}
        for group in ['Base+M2','Base+M1+M2']:
            ee=joblib.load(job/(group+'_selection_episodes.joblib'))
            sa.check_validation(ee,[c.id for c in val],[c.id for c in test]);cached[group]={e['cell_id']:e for e in ee}
        vp={}
        for c in val:
            vp[c.id]={'B':cached['Base+M2'][c.id]['base'],'B1':cached['Base+M1+M2'][c.id]['base']}
            for short,group in [('B2','Base+M2'),('B12','Base+M1+M2')]:
                s=entries[group]['selection'];vp[c.id][short]=cached[group][c.id]['reference_predictions'][s['reference_view']][s['reference_index']]
        choices={};trials=[]
        for group in GROUPS:
            scores=[]
            baseline_domain={r['domain']:r for r in aggregate([dict(dataset=c.dataset,domain=c.domain,
                **error_metrics(c.y[sa.K:],vp[c.id][group])) for c in val],['domain'])} if args.risk_constrained else {}
            for kind in P['candidates']:
                rr={c.id:correction(sa.inference_view(c),vp[c.id][group],kind,P['clip']) for c in val}
                for w in P['weights']:
                    loss=sa.macro([dict(dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(
                        vp[c.id][group]+w*rr[c.id]-c.y[sa.K:])))) for c in val])
                    eligible=True;worst=0.
                    if args.risk_constrained:
                        domain_rows=aggregate([dict(dataset=c.dataset,domain=c.domain,
                            **error_metrics(c.y[sa.K:],vp[c.id][group]+w*rr[c.id])) for c in val],['domain'])
                        worst=max(r[m]-baseline_domain[r['domain']][m] for r in domain_rows for m in ['mae','rmse','p95_ae'])
                        eligible=worst<=1e-10
                    if eligible:scores.append((loss,w,kind))
                    trials.append(dict(group=group,kind=kind,weight=w,validation_mae=loss,eligible=eligible,max_domain_metric_increase_pp=worst))
            loss,w,kind=min(scores);choices[group]=dict(kind=kind,weight=w,validation_mae=loss)
        sa.write_json(dest/'selection.json',dict(choices=choices,trials=trials,validation_cells=[c.id for c in val]))
        selections.append(dict(fold=fold,choices=choices))
        for c in test:
            pred={}
            for short,group in [('B2','Base+M2'),('B12','Base+M1+M2')]:
                with np.load(job/group/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:])
                    pred[short]=z[manifest['variant']].copy();pred['B' if short=='B2' else 'B1']=z['parent'].copy()
            for group,s in choices.items():
                p=pred[group];r=correction(sa.inference_view(c),p,s['kind'],P['clip'])
                changed=c.y.copy();changed[sa.K:]=123
                boundary=max(boundary,float(np.max(abs(r-correction(replace(c,y=changed),p,s['kind'],P['clip'])))))
                for n in sorted({sa.K+1,min(len(c.x),sa.K+17),max(sa.K+1,len(c.x)//2)}):
                    alt=correction(sa.prefix(c,n),p[:n-sa.K],s['kind'],P['clip'])
                    boundary=max(boundary,float(np.max(abs(r[:n-sa.K]-alt))))
                assert boundary<1e-8
                pred[group+'3']=p+s['weight']*r
            path=dest/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,y=c.y[sa.K:],**pred)
            with np.load(path) as z:
                for group,s in choices.items():
                    alt=z[group]+s['weight']*correction(sa.inference_view(c),z[group],s['kind'],P['clip'])
                    replay=max(replay,float(np.max(abs(alt-z[group+'3']))))
                for group in pred:rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=group,**error_metrics(z['y'],z[group])))
        print(fold,'complete',flush=True)
    assert len(rows)==2920 and replay<1e-9
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    edges=[('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]
    gates={a+'<'+b:all(look[ds,a][m]<look[ds,b][m] for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae']) for a,b in edges}
    pairs=[]
    for a,b in edges:
        for ds in ['XJTU','MATR','Tongji']:
            aa={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==a};bb={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==b}
            for m in ['mae','rmse','p95_ae']:
                pairs.append(dict(comparison=a+'-'+b,dataset=ds,metric=m,**paired_stats([(aa[c][m]-bb[c][m])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    reverse={v:k for k,v in GROUPS.items()}
    parent_error=max(abs(look[r['dataset'],reverse[r['group']]][m]-100*r[m]) for r in manifest['table'] for m in ['mae','rmse','p95_ae'])
    assert parent_error<1e-8 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',cells=365,folds=21,table=table,gates=gates,
        selections=selections,paired_comparisons=pairs,max_boundary_error=boundary,max_replay_error=replay,max_frozen_parent_metric_error=parent_error))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
