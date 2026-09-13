"""Independent replay of source OOF, risk costs and all stored candidate predictions."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,SCREEN,GROUPS,error_metrics,aggregate
from reestimated_screen import augmented,outputs,VIEW
from summarize_strict import paired_stats
import reference_gate as rg


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();root=Path(args.out)
    summary=json.loads((root/'summary.json').read_text());manifest=json.loads((FROZEN/'candidate.json').read_text())
    run=json.loads((root/'run_protocol.json').read_text());assert run['code_sha256']==sa.digest(HERE/'reestimated_screen.py')
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[];oof_error=0.;prediction_error=0.;risk_error=0.;fits=0
    with threadpool_limits(limits=1):
        for dest in sorted((root/'folds').iterdir()):
            fold=dest.name.replace('__',':',1);tr,val,test=sa.split(cells,fold);byid={c.id:c for c in tr}
            audit=json.loads((dest/'source_oof_audit.json').read_text());s=json.loads((dest/'selection.json').read_text())
            assert set(s['validation_cells'])=={c.id for c in val}
            budget={tuple(k) for k in audit['source_keys']};assert len(budget)<=1000
            fullpath=HERE.parent/'results/m3_progression_screen_v1/folds'/dest.name/(s['branch_key']+'.joblib')
            assert sa.digest(fullpath)==audit['full_branch_sha256'];branch=joblib.load(fullpath)
            oof=joblib.load(dest/'source_oof_predictions.joblib');seen=set()
            for a in audit['domains']:
                model=joblib.load(dest/('source_oof__'+a['domain']+'.joblib'));fit={tuple(k) for k in a['fit_keys']};held={tuple(k) for k in a['held_keys']}
                assert fit|held==budget and not fit&held and set(model.source_keys)==fit
                assert all(byid[cid].domain!=a['domain'] for cid,j in fit)
                assert all(byid[cid].domain==a['domain'] for cid,j in held)
                # Independently rebuild exact fitted target vector from ONLY stored source keys.
                target=np.asarray([byid[cid].y[j]-np.mean(byid[cid].y[:sa.K]) for cid,j in model.source_keys],dtype=np.float32)
                gp=model.parent.gp
                np.testing.assert_allclose((target-gp._y_train_mean)/gp._y_train_std,gp.y_train_,rtol=0,atol=1e-7)
                for c in tr:
                    if c.domain!=a['domain']:continue
                    idx=sorted(j for cid,j in held if cid==c.id and j>=sa.K)
                    pred=model.predict(sa.inference_view(c))[np.asarray(idx)-sa.K]
                    oof_error=max(oof_error,float(np.max(abs(pred-oof[c.id]))));seen.add(c.id)
                fits+=1
            assert seen==set(byid)
            job=SCREEN/dest.name/'seed_0';src=joblib.load(job/'source_lodo_episodes.joblib')
            entries={r['group']:r for r in manifest['manifest'] if r['fold']==fold};heads={};adapters={};oldheads={}
            for family,g in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
                heads[g]=joblib.load(dest/(g+'_new_head.joblib'));adapters[g]=joblib.load(FROZEN/entries[g]['artifact'])
                oldheads[g]=joblib.load(job/(g+'_reference_heads.joblib'))
                assert sa.digest(job/(g+'_reference_heads.joblib'))==entries[g]['screen_head_sha256']
                costs=np.asarray([[rg.expert_risk(augmented(e,oof[e['cell_id']],s['weight']),p) for p in heads[g]['params']] for e in src[family]])
                risk_error=max(risk_error,float(np.max(abs(costs-heads[g]['costs']))))
            for c in test:
                bp=branch.predict(sa.inference_view(c));pred={}
                for short,g,raw in [('B2','Base+M2','B'),('B12','Base+M1+M2','B1')]:
                    a=adapters[g];base,r=sa.predict_components(a.parent,c,a.parent_mode)
                    phy=a.physical.predict(sa.inference_view(c),a.physical_mode)
                    ep=dict(base=base,residual=r,physical=phy)
                    pred[raw]=base;pred[short]=outputs(ep,c,oldheads[g])[a.selection['reference_index']]
                    newep=augmented(ep,bp,s['weight']);pred[raw+'3']=newep['base']
                    pred[short+'3']=outputs(newep,c,heads[g])[s['settings'][g]['index']]
                    pred[short+'3_old_risk']=outputs(newep,c,oldheads[g])[a.selection['reference_index']]
                with np.load(dest/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:])
                    for g,p in pred.items():
                        prediction_error=max(prediction_error,float(np.max(abs(p-z[g]))))
                        rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(z['y'],z[g])))
            print(fold,'OOF/risk/full replay audited',flush=True)
    assert len(rows)==3650 and max(oof_error,risk_error,prediction_error)<1e-8
    table=aggregate(rows,['dataset','group']);lookup={(r['dataset'],r['group']):r for r in table}
    metric_error=max(abs(lookup[r['dataset'],r['group']][m]-r[m]) for r in summary['table'] for m in ['mae','rmse','p95_ae'])
    assert metric_error<1e-10
    pairs=[]
    for a,b in [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B123','B123_old_risk')]:
        for ds in ['XJTU','MATR','Tongji']:
            aa={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==a};bb={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==b}
            for m in ['mae','rmse','p95_ae']:pairs.append(dict(comparison=a+'-'+b,dataset=ds,metric=m,**paired_stats(
                [(aa[c][m]-bb[c][m])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    sa.write_json(root/'verification.json',dict(status='PASS',source_oof_fits=fits,rows=len(rows),max_source_oof_error=oof_error,
        max_risk_cost_error=risk_error,max_full_prediction_error=prediction_error,max_metric_error=metric_error,paired_comparisons=pairs,
        limits=['Development evidence, not independent confirmation or novelty proof.']))
    print('PASS',fits,'source OOF fits',prediction_error,flush=True)


if __name__=='__main__':main()
