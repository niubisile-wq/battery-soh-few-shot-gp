"""Source-budget matched M3 OOF and same-algorithm M2 risk-table re-estimation."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,SCREEN,GROUPS,aggregate,error_metrics,dump_csv
from progression import ProgressionGP
from conditional_geometry import ConditionalGeometry
import reference_gate as rg

P=json.loads((HERE/'reestimated_protocol.json').read_text())
VIEW='hi_pca8_bagged'


def augmented(e,branch,w):
    return dict(e,base=(1-w)*e['base']+w*branch)


def outputs(e,c,head):return rg.attach(e,sa.inference_view(c),head)['reference_predictions'][VIEW]


def run_fold(fold,out):
    with threadpool_limits(limits=1):
        cells=sa.load_cells(fold.split(':')[0]);tr,va,te=sa.split(cells,fold)
        byid={c.id:c for c in tr};name=fold.replace(':','__');dest=Path(out)/'folds'/name;dest.mkdir(parents=True,exist_ok=False)
        job=SCREEN/name/'seed_0';manifest=json.loads((FROZEN/'candidate.json').read_text())
        entries={r['group']:r for r in manifest['manifest'] if r['fold']==fold}
        chosen=json.loads((HERE.parent/'results/m3_shared_progression_screen_v1/folds'/name/'selection.json').read_text())['choices']['full']
        key=chosen['key'];view,kernel,mode=key.split('__')
        branch_path=HERE.parent/'results/m3_progression_screen_v1/folds'/name/(key+'.joblib');full_branch=joblib.load(branch_path)
        allowed=defaultdict(list)
        for cid,j in entries['Base+M2']['source_keys']:allowed[cid].append(j)
        budget={(cid,j) for cid,idx in allowed.items() for j in idx};assert set(allowed)==set(byid)
        assert set(full_branch.source_keys)==budget and len(budget)<=1000
        source=joblib.load(job/'source_lodo_episodes.joblib');audits=json.loads((job/'source_lodo_audit.json').read_text())
        oof={};new_audits=[]
        for audit in audits:
            domain=audit['domain'];fit={tuple(k) for k in audit['fit_keys']};held={tuple(k) for k in audit['held_keys']}
            assert fit|held==budget and not fit&held
            assert all(byid[cid].domain!=domain for cid,j in fit)
            assert all(byid[cid].domain==domain for cid,j in held)
            fitcells=[c for c in tr if c.domain!=domain];fit_allowed={c.id:allowed[c.id] for c in fitcells}
            gp=ProgressionGP(view,kernel).fit([sa.source_view(c,fit_allowed[c.id]) for c in fitcells],fit_allowed)
            assert set(gp.source_keys)==fit
            model=ConditionalGeometry(gp,mode);joblib.dump(model,dest/('source_oof__'+domain+'.joblib'))
            for c in tr:
                if c.domain!=domain:continue
                idx=np.asarray([j for j in allowed[c.id] if j>=sa.K]);q=model.predict(sa.inference_view(c))
                oof[c.id]=q[idx-sa.K]
                assert {(c.id,int(j)) for j in idx}<={tuple(k) for k in audit['query_keys']}
            new_audits.append(dict(domain=domain,fit_keys=sorted(fit),held_keys=sorted(held)))
        assert set(oof)==set(byid)
        for family in ['Base','Base+M1']:
            assert {e['cell_id'] for e in source[family]}==set(byid)
            for e in source[family]:
                c=byid[e['cell_id']];idx=[j for j in allowed[c.id] if j>=sa.K]
                assert np.array_equal(e['y'],c.y[idx]) and len(oof[c.id])==len(idx)
        joblib.dump(oof,dest/'source_oof_predictions.joblib')
        sa.write_json(dest/'source_oof_audit.json',dict(branch_key=key,full_branch_sha256=sa.digest(branch_path),source_keys=sorted(budget),domains=new_audits))
        print(fold,'new branch source OOF complete',flush=True)
        ve={};old_heads={}
        for g in ['Base+M2','Base+M1+M2']:
            ee=joblib.load(job/(g+'_selection_episodes.joblib'));sa.check_validation(ee,[c.id for c in va],[c.id for c in te])
            ve[g]={e['cell_id']:e for e in ee};old_heads[g]=joblib.load(job/(g+'_reference_heads.joblib'))
            assert sa.digest(job/(g+'_reference_heads.joblib'))==entries[g]['screen_head_sha256']
        vb={c.id:full_branch.predict(sa.inference_view(c)) for c in va};heads={};trials=[];choices_by_weight={};zero_error=0.
        for w in P['weights']:
            selections={}
            for family,g in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
                augmented_source=[augmented(e,oof[e['cell_id']],w) for e in source[family]]
                head=rg.fit(augmented_source,[sa.source_view(c,allowed[c.id]) for c in tr],metric=True,bagged=True)
                heads[w,g]=head;pp={c.id:outputs(augmented(ve[g][c.id],vb[c.id],w),c,head) for c in va}
                if w==0:
                    old_index=entries[g]['selection']['reference_index']
                    for c in va:
                        expected=ve[g][c.id]['reference_predictions'][VIEW]
                        zero_error=max(zero_error,float(np.max(abs(pp[c.id]-expected))))
                    assert zero_error<1e-8
                    settings=[old_index]
                else:settings=range(len(head['settings']))
                scores=[]
                for index in settings:
                    loss=sa.macro([dict(dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(pp[c.id][index]-c.y[sa.K:])))) for c in va])
                    scores.append((loss,index));trials.append(dict(weight=w,group=g,index=index,validation_mae=loss))
                loss,index=min(scores);selections[g]=dict(index=index,validation_mae=loss)
            choices_by_weight[w]=selections
        _,w=min((s['Base+M1+M2']['validation_mae'],w) for w,s in choices_by_weight.items())
        chosen=choices_by_weight[w]
        sa.write_json(dest/'selection.json',dict(branch_key=key,weight=w,settings=chosen,trials=trials,validation_cells=[c.id for c in va],zero_weight_replay_error=zero_error))
        for g in ['Base+M2','Base+M1+M2']:joblib.dump(heads[w,g],dest/(g+'_new_head.joblib'))
        rows=[];boundary=0.
        for c in te:
            branch=full_branch.predict(sa.inference_view(c));n=min(len(c.x),sa.K+3);yy=c.y.copy();yy[sa.K:]=123
            boundary=max(boundary,float(np.max(abs(branch-full_branch.predict(replace(c,y=yy))))),
                float(np.max(abs(branch[:n-sa.K]-full_branch.predict(sa.prefix(c,n))))))
            pred={}
            for short,g,raw in [('B2','Base+M2','B'),('B12','Base+M1+M2','B1')]:
                with np.load(job/g/(c.id+'.npz')) as z:
                    pred[short]=z[manifest['variant']].copy();pred[raw]=z['parent'].copy()
                # Need frozen support residual/physical predictions, not query truth.
                adapter=joblib.load(FROZEN/entries[g]['artifact'])
                base,residual=sa.predict_components(adapter.parent,c,adapter.parent_mode)
                physical=adapter.physical.predict(sa.inference_view(c),adapter.physical_mode)
                assert np.max(abs(base-pred[raw]))<1e-8
                e=dict(base=(1-w)*base+w*branch,residual=residual,physical=physical)
                new=outputs(e,c,heads[w,g])[chosen[g]['index']]
                pred[raw+'3']=e['base'];pred[short+'3']=new
                old_index=entries[g]['selection']['reference_index']
                pred[short+'3_old_risk']=outputs(e,c,old_heads[g])[old_index]
                altered=outputs(e,replace(c,y=yy),heads[w,g])[chosen[g]['index']]
                boundary=max(boundary,float(np.max(abs(new-altered))))
            assert boundary<1e-8
            path=dest/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=c.y[sa.K:],**pred)
            for group,p in pred.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=group,**error_metrics(c.y[sa.K:],p)))
        sa.write_json(dest/'result.json',dict(fold=fold,rows=rows,max_boundary_error=boundary,weight=w,branch_key=key))
        print(fold,'risk re-estimation and ablation complete',flush=True);return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((FROZEN/'candidate.json').read_text());assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    sa.write_json(out/'run_protocol.json',dict(protocol=P,code_sha256=sa.digest(Path(__file__)),protocol_sha256=sa.digest(HERE/'reestimated_protocol.json')))
    rows=[];folds=sorted({r['fold'] for r in manifest['manifest']})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in as_completed([pool.submit(run_fold,fold,str(out)) for fold in folds]):rows.extend(task.result())
    assert len(rows)==365*10
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    gates={a+'<'+b:all(look[ds,a][m]<look[ds,b][m] for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae'])
        for a,b in [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]}
    reverse={v:k for k,v in GROUPS.items()};err=max(abs(look[r['dataset'],reverse[r['group']]][m]-100*r[m]) for r in manifest['table'] for m in ['mae','rmse','p95_ae'])
    assert err<1e-8 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,cells=365,folds=21,max_frozen_parent_metric_error=err))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
