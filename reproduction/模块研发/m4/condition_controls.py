"""Separately validation-selected raw and residualized geometry controls."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=HERE.parent/'results';protocol=json.loads((HERE/'condition_controls_protocol.json').read_text())
    batch=json.loads((root/'m4_condition_batch_v1/result.json').read_text());assert batch['status']=='VALIDATION_BATCH_COMPLETE'
    selections=[]
    for e in batch['folds']:
        folder=Path(e['correction']);r=json.loads((folder/'result.json').read_text());choices={}
        for family in protocol['families']:
            choice=min([t for t in r['trials'] if t['key'].startswith(family+'__')],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
            paths={g:folder/'models'/(g+'__'+choice['key']+'.joblib') for g in PARENTS}
            choices[family]=dict(choice=choice,models={g:dict(path=str(p),sha256=sa.digest(p)) for g,p in paths.items()})
        selections.append(dict(fold=e['fold'],choices=choices))
    sa.write_json(out/'frozen_selections.json',dict(selections=selections,protocol=protocol,code_sha256=sa.digest(Path(__file__))))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];candidate=root/'m3_waveform_candidate_v1';rows=[]
    with threadpool_limits(limits=1):
        for e in selections:
            fold=e['fold'];name=fold.replace(':','__');_,_,test=sa.split(cells,fold)
            models={family:{g:joblib.load(p['path']) for g,p in s['models'].items()} for family,s in e['choices'].items()}
            for c in test:
                with np.load(candidate/'folds'/name/'predictions'/(c.id+'.npz')) as z:y=z['y'];pp={g:z[g] for g in PARENTS}
                assert np.array_equal(y,c.y[sa.K:])
                for family,mm in models.items():
                    gain=e['choices'][family]['choice']['gain']
                    for g,m in mm.items():pp[g+'4__'+family]=pp[g]+gain*m.predict(sa.inference_view(c),pp[g])
                path=out/'folds'/name/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,**pp)
                for g,p in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,p)))
            print(fold,'both geometry families scored',flush=True)
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={}
    for family in protocol['families']:
        tag=lambda g:g+'__'+family if '4' in g else g
        gates[family]={a+'<'+b:all(look[ds,tag(a)][k]<look[ds,tag(b)][k] for ds in ['XJTU','MATR','Tongji'] for k in ['mae','rmse','p95_ae']) for a,b in addition_edges()}
    assert len(rows)==8760;dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='CONTROLS_COMPLETE_NOT_ADOPTED',table=table,gates=gates))


if __name__=='__main__':main()
