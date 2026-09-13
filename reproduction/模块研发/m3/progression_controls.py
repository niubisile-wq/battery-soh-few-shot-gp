"""Separately tuned declared progression-only and joint-representation controls."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,GROUPS,aggregate,error_metrics,dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    root=Path(args.root).resolve();out=root/'matched_controls';out.mkdir(exist_ok=False)
    run=json.loads((root/'run_protocol.json').read_text())
    assert run['protocol']['phase']=='Causal progression and reference-relative charge geometry GP screen v1'
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[];selections=[]
    with threadpool_limits(limits=1):
        for dest in sorted((root/'folds').iterdir()):
            fold=dest.name.replace('__',':',1);_,val,test=sa.split(cells,fold)
            old=json.loads((dest/'selection.json').read_text())
            assert set(old['validation_cells'])=={c.id for c in val}
            choices={}
            for family in ['progress_only','joint']:
                choices[family]={}
                for group in GROUPS:
                    trials=[r for r in old['trials'] if r['group']==group and
                        r['key'].startswith('progress_only__')==(family=='progress_only')]
                    s=min(trials,key=lambda r:(r['validation_mae'],r['weight'],r['key']))
                    choices[family][group]=s
            # Only stored validation scores determine choices; test files are unopened above.
            sa.write_json(out/'folds'/dest.name/'selection.json',choices)
            selections.append(dict(fold=fold,choices=choices))
            models={key:joblib.load(dest/(key+'.joblib')) for key in {s['key'] for cc in choices.values() for s in cc.values()}}
            for c in test:
                with np.load(dest/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:])
                    branch={key:model.predict(sa.inference_view(c)) for key,model in models.items()}
                    pred={g:z[g] for g in GROUPS}
                    for family,cc in choices.items():
                        for group,s in cc.items():
                            name=group+'3__'+family;w=s['weight'];pred[name]=(1-w)*z[group]+w*branch[s['key']]
                    path=out/'folds'/dest.name/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,y=z['y'],**pred)
                    for group,p in pred.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=group,**error_metrics(z['y'],p)))
            print(fold,'matched controls complete',flush=True)
    assert len(rows)==365*12
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    gates={}
    for family in ['progress_only','joint']:
        gates[family]={a+'<'+b:all(look[ds,a+'__'+family][m]<look[ds,b if b in GROUPS else b+'__'+family][m]
            for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae'])
            for a,b in [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3')]}
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='CONTROLS_COMPLETE_NOT_ADOPTED',table=table,gates=gates,selections=selections,
        limits=['Exploratory controls from prospectively declared feature families.', 'No choice based on test scores.', 'Not sufficient novelty proof or independent confirmation.']))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
