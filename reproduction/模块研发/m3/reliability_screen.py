"""Error-trained gate screen reusing fully audited source-LODO branch predictions."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,SCREEN,GROUPS,aggregate,error_metrics,dump_csv
from reliability import ReliabilityGate
from summarize_strict import paired_stats

P=json.loads((HERE/'reliability_protocol.json').read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);source_root=HERE.parent/'results'/P['source_oof_run']
    assert json.loads((source_root/'verification.json').read_text())['status']=='PASS'
    manifest=json.loads((FROZEN/'candidate.json').read_text());assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    sa.write_json(out/'run_protocol.json',dict(protocol=P,hashes={p.name:sa.digest(p) for p in [HERE/'reliability.py',HERE/'reliability_protocol.json',Path(__file__)]}))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[];selections=[];boundary=0.
    with threadpool_limits(limits=1):
        for fold in sorted({r['fold'] for r in manifest['manifest']}):
            tr,val,test=sa.split(cells,fold);name=fold.replace(':','__');dest=out/'folds'/name;dest.mkdir(parents=True)
            audit=json.loads((source_root/'folds'/name/'source_oof_audit.json').read_text())
            branch_path=HERE.parent/'results/m3_progression_screen_v1/folds'/name/(audit['branch_key']+'.joblib')
            assert sa.digest(branch_path)==audit['full_branch_sha256'];branch_model=joblib.load(branch_path)
            allowed=defaultdict(list)
            for cid,j in audit['source_keys']:allowed[cid].append(j)
            assert set(allowed)=={c.id for c in tr}
            source=joblib.load(SCREEN/name/'seed_0/source_lodo_episodes.joblib');oof=joblib.load(source_root/'folds'/name/'source_oof_predictions.joblib')
            models={}
            for family in ['B','B1']:
                for view in P['views']:
                    for alpha in P['alphas']:
                        key=view+'__'+str(alpha)
                        models[family,key]=ReliabilityGate(view,alpha).fit(source[GROUPS[family]],oof,
                            [sa.source_view(c,allowed[c.id]) for c in tr],allowed)
                        joblib.dump(models[family,key],dest/(family+'__'+key+'.joblib'))
            entry=next(r for r in manifest['manifest'] if r['fold']==fold and r['group']=='Base+M1+M2')
            ee=joblib.load(SCREEN/name/'seed_0/Base+M1+M2_selection_episodes.joblib');sa.check_validation(ee,[c.id for c in val],[c.id for c in test])
            cached={e['cell_id']:e for e in ee};s=entry['selection']
            vp={c.id:cached[c.id]['reference_predictions'][s['reference_view']][s['reference_index']] for c in val}
            vb={c.id:branch_model.predict(sa.inference_view(c)) for c in val};scores=[];trials=[]
            for key in sorted({k for family,k in models}):
                gate=models['B1',key];weights={c.id:gate.predict(sa.inference_view(c),vp[c.id],vb[c.id]) for c in val}
                for gain in P['gains']:
                    loss=sa.macro([dict(dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(
                        vp[c.id]+gain*weights[c.id]*(vb[c.id]-vp[c.id])-c.y[sa.K:])))) for c in val])
                    scores.append((loss,gain,key));trials.append(dict(key=key,gain=gain,validation_mae=loss))
            loss,gain,key=min(scores);choice=dict(key=key,gain=gain,validation_mae=loss,branch_path=str(branch_path),branch_sha256=sa.digest(branch_path))
            sa.write_json(dest/'selection.json',dict(chosen=choice,trials=trials,validation_cells=[c.id for c in val],source_keys=audit['source_keys']))
            selections.append(dict(fold=fold,chosen=choice))
            for c in test:
                b=branch_model.predict(sa.inference_view(c));parent={}
                for g,raw in [('Base+M2','B'),('Base+M1+M2','B1')]:
                    with np.load(SCREEN/name/'seed_0'/g/(c.id+'.npz')) as z:
                        assert np.array_equal(z['y'],c.y[sa.K:]);parent[raw]=z['parent'].copy();parent['B2' if raw=='B' else 'B12']=z[manifest['variant']].copy()
                pp=dict(parent);weights={};yy=c.y.copy();yy[sa.K:]=123;n=min(len(c.x),sa.K+3)
                b2=branch_model.predict(replace(c,y=yy));bs=branch_model.predict(sa.prefix(c,n))
                for g in GROUPS:
                    family='B1' if '1' in g else 'B';gate=models[family,key];w=gate.predict(sa.inference_view(c),parent[g],b)
                    alt=gate.predict(replace(c,y=yy),parent[g],b2);short=gate.predict(sa.prefix(c,n),parent[g][:n-sa.K],bs)
                    boundary=max(boundary,float(np.max(abs(w-alt))),float(np.max(abs(w[:n-sa.K]-short))),float(np.max(abs(b-b2))),float(np.max(abs(b[:n-sa.K]-bs))))
                    pp[g+'3']=parent[g]+gain*w*(b-parent[g]);pp[g+'3_constant']=parent[g]+gain*gate.constant*(b-parent[g]);weights[g]=w
                assert boundary<1e-8
                path=dest/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,y=c.y[sa.K:],branch=b,**pp,**{g+'_weight':w for g,w in weights.items()})
                for g,p in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(c.y[sa.K:],p)))
            print(fold,'bounded reliability gate complete',flush=True)
    assert len(rows)==4380
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    edges=[('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]
    gates={a+'<'+b:all(look[ds,a][m]<look[ds,b][m] for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae']) for a,b in edges}
    reverse={v:k for k,v in GROUPS.items()};err=max(abs(look[r['dataset'],reverse[r['group']]][m]-100*r[m]) for r in manifest['table'] for m in ['mae','rmse','p95_ae'])
    assert err<1e-8 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    pairs=[]
    for a,b in edges+[('B123','B123_constant')]:
        for ds in ['XJTU','MATR','Tongji']:
            aa={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==a};bb={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==b}
            for m in ['mae','rmse','p95_ae']:pairs.append(dict(comparison=a+'-'+b,dataset=ds,metric=m,**paired_stats(
                [(aa[c][m]-bb[c][m])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,selections=selections,paired_comparisons=pairs,
        max_boundary_error=boundary,max_frozen_parent_metric_error=err,cells=365,folds=21))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
