"""Structured branch kernel-family selections and same-config kernel swap."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from conditional_geometry import ConditionalGeometry
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def main():
    results=HERE.parent/'results';batch=json.loads((results/'m4_structured_batch_v1/result.json').read_text())
    assert batch['status']=='VALIDATION_BATCH_COMPLETE' and len(batch['folds'])==21
    out=results/'m4_structured_controls_v1';out.mkdir(exist_ok=False);selections=[]
    for e in batch['folds']:
        root=Path(e['branch']);r=json.loads((root/'result.json').read_text())
        audit=json.loads((root/'verification.json').read_text());assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
        choices={}
        for kernel in ('joint','additive'):
            choices[kernel]=min([t for t in r['trials'] if t['key'].split('__')[1]==kernel],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
        original=r['selected'];enc,kernel,mode=original['key'].split('__')
        choices['kernel_swap']=dict(key='__'.join((enc,'joint' if kernel=='additive' else 'additive',mode)),gain=original['gain'])
        specs={}
        for label,choice in choices.items():
            enc,kernel,mode=choice['key'].split('__')
            spec=next(m for m in r['models'] if (m['encoder'],m['kernel'])==(enc,kernel))
            assert sa.digest(Path(spec['path']))==spec['sha256']
            specs[label]=dict(choice=choice,model=spec,mode=mode)
        selections.append(dict(fold=e['fold'],controls=specs,source_result_sha256=sa.digest(root/'result.json')))
    sa.write_json(out/'frozen_selections.json',dict(selections=selections,code_sha256=sa.digest(Path(__file__)),
        rule='Joint/additive each separately validation-selected; kernel swap holds original mixed-selected encoder,adaptationmode andgain fixed. Report all controls, no datasetwise mixing.'))
    cells=[c for ds in ('XJTU','MATR','Tongji') for c in sa.load_cells(ds)];rows=[]
    with threadpool_limits(limits=1):
        for s in selections:
            models={label:ConditionalGeometry(joblib.load(spec['model']['path']),spec['mode']) for label,spec in s['controls'].items()}
            _,_,test=sa.split(cells,s['fold'])
            for c in test:
                tail=Path('folds')/s['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(results/'m3_waveform_candidate_v1'/tail) as z:y=z['y'];pp={g:z[g] for g in PARENTS}
                assert np.array_equal(y,c.y[sa.K:])
                for label,model in models.items():
                    q=model.predict(sa.inference_view(c));gain=s['controls'][label]['choice']['gain']
                    for g in PARENTS:pp[g+'4__'+label]=pp[g]+gain*(q-pp[g])
                path=out/tail;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,**pp)
                for g,pred in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,pred)))
            print(s['fold'],'three kernel controls scored',flush=True)
    assert len(rows)==11680
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={}
    for label in ('joint','additive','kernel_swap'):
        tag=lambda g:g+'__'+label if '4' in g else g
        gates[label]={a+'<'+b:all(look[ds,tag(a)][k]<look[ds,tag(b)][k] for ds in ('XJTU','MATR','Tongji') for k in ('mae','rmse','p95_ae')) for a,b in addition_edges()}
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='CONTROLS_COMPLETE_NOT_ADOPTED',rows=len(rows),table=table,gates=gates,
        limits='No independent output/metric audit yet. Reused development cohorts,not independent confirmation; additive kernel has more parameters.'))


if __name__=='__main__':main()
