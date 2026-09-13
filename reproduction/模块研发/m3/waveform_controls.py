"""Factorial waveform controls with shared full-model validation selection."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,GROUPS,aggregate,error_metrics,dump_csv

FAMILIES=['wave_absolute_pca8','wave_absolute_pls4','wave_pair_pca8','wave_pair_pls4']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root)
    out=root/'matched_controls';out.mkdir(exist_ok=False);cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)]
    rows=[];selections=[]
    with threadpool_limits(limits=1):
        for dest in sorted((root/'folds').iterdir()):
            fold=dest.name.replace('__',':',1);_,val,test=sa.split(cells,fold)
            old=json.loads((dest/'selection.json').read_text());assert set(old['validation_cells'])=={c.id for c in val}
            choices={family:min([r for r in old['trials'] if r['group']=='B12' and r['key'].startswith(family+'__')],
                key=lambda r:(r['validation_mae'],r['weight'],r['key'])) for family in FAMILIES}
            sa.write_json(out/'folds'/dest.name/'selection.json',choices);selections.append(dict(fold=fold,choices=choices))
            models={key:joblib.load(dest/(key+'.joblib')) for key in {s['key'] for s in choices.values()}}
            for c in test:
                b={key:m.predict(sa.inference_view(c)) for key,m in models.items()}
                with np.load(dest/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:]);pred={g:z[g] for g in GROUPS}
                    for family,s in choices.items():
                        w=s['weight']
                        for g in GROUPS:pred[g+'3__'+family]=(1-w)*z[g]+w*b[s['key']]
                    path=out/'folds'/dest.name/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=z['y'],**pred)
                    for g,p in pred.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(z['y'],p)))
            print(fold,'waveform controls complete',flush=True)
    assert len(rows)==365*20
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={}
    for family in FAMILIES:
        gates[family]={a+'<'+b:all(look[ds,a+'__'+family][m]<look[ds,b if b in GROUPS else b+'__'+family][m]
            for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae'])
            for a,b in [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3')]}
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='CONTROLS_COMPLETE_NOT_ADOPTED',table=table,gates=gates,selections=selections,
        limits=['Validation-only selected factorial controls; not proof of novelty or independent performance.']))
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
