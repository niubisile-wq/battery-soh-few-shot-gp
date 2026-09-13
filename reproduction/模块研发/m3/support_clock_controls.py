"""Matched shared-selection clock-only, joint and ordinary early-trend controls."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,GROUPS,SCREEN,FROZEN,aggregate,error_metrics,dump_csv
from support_clock import ordinary_trend


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    root=Path(args.root);out=root/'matched_controls';out.mkdir(exist_ok=False)
    run=json.loads((root/'run_protocol.json').read_text());weights=run['protocol']['weights']
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)]
    manifest=json.loads((FROZEN/'candidate.json').read_text());rows=[];selections=[]
    with threadpool_limits(limits=1):
        for dest in sorted((root/'folds').iterdir()):
            fold=dest.name.replace('__',':',1);_,val,test=sa.split(cells,fold)
            s=json.loads((dest/'selection.json').read_text());assert set(s['validation_cells'])=={c.id for c in val}
            choices={}
            for family in ['clock_only','joint']:
                trial=[r for r in s['trials'] if r['group']=='B12' and r['key'].startswith('clock_only__')==(family=='clock_only')]
                best=min(trial,key=lambda r:(r['validation_mae'],r['weight'],r['key']))
                choices[family]=best
            entry=next(r for r in manifest['manifest'] if r['fold']==fold and r['group']=='Base+M1+M2')
            e=joblib.load(SCREEN/dest.name/'seed_0/Base+M1+M2_selection_episodes.joblib')
            sa.check_validation(e,[c.id for c in val],[c.id for c in test]);byid={r['cell_id']:r for r in e};sel=entry['selection']
            vp={c.id:byid[c.id]['reference_predictions'][sel['reference_view']][sel['reference_index']] for c in val}
            trend={c.id:ordinary_trend(sa.inference_view(c)) for c in val}
            loss,w=min((sa.macro([dict(dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(
                (1-w)*vp[c.id]+w*trend[c.id]-c.y[sa.K:])))) for c in val]),w) for w in weights)
            choices['ordinary_trend']=dict(weight=w,validation_mae=loss)
            sa.write_json(out/'folds'/dest.name/'selection.json',choices);selections.append(dict(fold=fold,choices=choices))
            models={key:joblib.load(dest/(key+'.joblib')) for key in {s['key'] for s in choices.values() if 'key' in s}}
            for c in test:
                branch={key:model.predict(sa.inference_view(c)) for key,model in models.items()};trend=ordinary_trend(sa.inference_view(c))
                with np.load(dest/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:]);pred={g:z[g] for g in GROUPS}
                    for family,s in choices.items():
                        b=trend if family=='ordinary_trend' else branch[s['key']];w=s['weight']
                        for g in GROUPS:pred[g+'3__'+family]=(1-w)*z[g]+w*b
                    path=out/'folds'/dest.name/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=z['y'],**pred)
                    for g,p in pred.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(z['y'],p)))
            print(fold,'clock controls complete',flush=True)
    assert len(rows)==365*16
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={}
    for family in ['clock_only','joint','ordinary_trend']:
        gates[family]={a+'<'+b:all(look[ds,a+'__'+family][m]<look[ds,b if b in GROUPS else b+'__'+family][m]
            for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae'])
            for a,b in [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3')]}
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='CONTROLS_COMPLETE_NOT_ADOPTED',table=table,gates=gates,selections=selections))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
