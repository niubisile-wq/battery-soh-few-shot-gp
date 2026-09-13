"""Same-config/gain plain residual controls for anchored or centered correction."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from score import PARENTS
from evaluate import aggregate,error_metrics,dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--variant',choices=['anchored','centered'],default='anchored');args=ap.parse_args();root=Path(args.root).resolve()
    assert (root/'summary.json').exists()
    out=root/'matched_plain_controls';out.mkdir(exist_ok=False)
    selections=json.loads((root/'frozen_selections.json').read_text())['selections']
    original=json.loads((HERE.parent/'results/m4_source_batch_v1/result.json').read_text())
    folders={r['fold']:Path(r['correction']) for r in original['folds']};manifest=[]
    for s in selections:
        paths={g:folders[s['fold']]/'models'/(g+'__'+s['selected']['key']+'.joblib') for g in PARENTS}
        manifest.append(dict(fold=s['fold'],key=s['selected']['key'],gain=s['selected']['gain'],
            models={g:dict(path=str(p),sha256=sa.digest(p)) for g,p in paths.items()}))
    sa.write_json(out/'protocol.json',dict(controls=manifest,comparison=f'Plain and {args.variant} residuals with identical feature family, learner, alpha and gain; separately fitted on same source OOF.',
        limit=f'Configuration selected for {args.variant} family; also report separately validation-selected first-screen family.'))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[]
    with threadpool_limits(limits=1):
        for s in manifest:
            _,_,test=sa.split(cells,s['fold']);models={g:joblib.load(v['path']) for g,v in s['models'].items()}
            for c in test:
                tail=Path('folds')/s['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(root/tail) as z:
                    pp={}
                    for g in PARENTS:
                        pp[g+'4_'+args.variant]=z[g+'4']
                        pp[g+'4_plain_matched']=z[g]+s['gain']*models[g].predict(sa.inference_view(c),z[g])
                    p=out/tail;p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,y=z['y'],**pp)
                    for g,pred in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(z['y'],pred)))
            print(s['fold'],'matched plain controls scored',flush=True)
    assert len(rows)==5840
    table=aggregate(rows,['dataset','group']);dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='CONTROLS_COMPLETE',table=table,rows=len(rows),code_sha256=sa.digest(Path(__file__))))


if __name__=='__main__':main()
