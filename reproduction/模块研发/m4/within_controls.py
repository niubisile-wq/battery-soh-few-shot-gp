"""Same-key/gain controls for within-domain source fitting, no re-selection."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa, HERE
from score import PARENTS
from evaluate import aggregate, error_metrics, dump_csv


def main():
    results = HERE.parent / 'results'; root = results / 'm4_within_screen_v1'
    assert (root / 'summary.json').exists()
    out = root / 'matched_controls'; out.mkdir(exist_ok=False)
    selections = json.loads((root / 'frozen_selections.json').read_text())['selections']
    folders = {}
    for label, directory in [('plain','m4_source_batch_v1'), ('anchored','m4_anchored_batch_v1'), ('centered','m4_centered_batch_v1')]:
        batch = json.loads((results / directory / 'result.json').read_text())
        assert batch['status']=='VALIDATION_BATCH_COMPLETE'
        folders[label] = {r['fold']: Path(r['correction']) for r in batch['folds']}
    manifest=[]
    for s in selections:
        models={}
        for label, ff in folders.items():
            paths={g:ff[s['fold']]/'models'/(g+'__'+s['selected']['key']+'.joblib') for g in PARENTS}
            models[label]={g:dict(path=str(p),sha256=sa.digest(p)) for g,p in paths.items()}
        manifest.append(dict(fold=s['fold'],gain=s['selected']['gain'],key=s['selected']['key'],models=models))
    sa.write_json(out/'protocol.json',dict(selections=manifest,code_sha256=sa.digest(Path(__file__)),
        purpose='Same key/gain across within,anchored,centered,plain. No re-selection or datasetwise combination.',
        limits='Configuration selected for within candidate; standard controls are not independent confirmation.'))
    cells=[c for ds in ('XJTU','MATR','Tongji') for c in sa.load_cells(ds)];rows=[]
    with threadpool_limits(limits=1):
        for s in manifest:
            mm={}
            for label,models in s['models'].items():
                mm[label]={}
                for g,spec in models.items():
                    assert sa.digest(Path(spec['path']))==spec['sha256']
                    mm[label][g]=joblib.load(spec['path'])
            _,_,test=sa.split(cells,s['fold'])
            for c in test:
                tail=Path('folds')/s['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(root/tail) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:])
                    pp={g+'4_within':z[g+'4'] for g in PARENTS}
                    for label,models in mm.items():
                        for g,m in models.items():pp[g+'4_'+label]=z[g]+s['gain']*m.predict(sa.inference_view(c),z[g])
                    path=out/tail;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=z['y'],**pp)
                    for g,pred in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(z['y'],pred)))
            print(s['fold'],'three matched controls scored',flush=True)
    assert len(rows)==11680
    table=aggregate(rows,['dataset','group']);dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='CONTROLS_COMPLETE_NOT_ADOPTED',rows=len(rows),table=table,
        limits='Shared score calculator, not an independent prediction/metric audit or generalization claim.'))


if __name__=='__main__':main()
