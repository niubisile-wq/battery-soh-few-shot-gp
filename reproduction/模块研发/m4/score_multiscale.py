"""Score frozen21 direct-branch selections, all16 ablations plus standalone branch."""
from dataclasses import replace
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN
from conditional_geometry import ConditionalGeometry
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def main():
    ap=argparse.ArgumentParser();variants=ap.add_mutually_exclusive_group();variants.add_argument('--structured',action='store_true');variants.add_argument('--registered',action='store_true');args=ap.parse_args()
    variant='registered' if args.registered else 'structured' if args.structured else 'multiscale'
    results=HERE.parent/'results';batchroot=results/('m4_'+variant+'_batch_v1')
    batch=json.loads((batchroot/'result.json').read_text())
    assert batch['status']=='VALIDATION_BATCH_COMPLETE' and len(batch['folds'])==21
    out=results/('m4_'+variant+'_screen_v1');out.mkdir(exist_ok=False)
    candidate=results/'m3_waveform_candidate_v1';parent=json.loads((candidate/'candidate.json').read_text())
    old=json.loads((FROZEN/'candidate.json').read_text());selections=[]
    for e in batch['folds']:
        root=Path(e['branch']);r=json.loads((root/'result.json').read_text())
        audit=json.loads((root/'verification.json').read_text())
        assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
        assert audit['code_sha256']==sa.digest(HERE/'audit_multiscale.py')
        choice=r['selected'];assert choice==min(r['trials'],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
        enc,kernel,mode=choice['key'].split('__')
        spec=next(m for m in r['models'] if (m['encoder'],m['kernel'])==(enc,kernel))
        assert sa.digest(Path(spec['path']))==spec['sha256']
        selections.append(dict(fold=e['fold'],branch_root=str(root),selected=choice,model=spec,mode=mode,
                               result_sha256=sa.digest(root/'result.json')))
    sa.write_json(out/'frozen_selections.json',dict(selections=selections,
        parent_manifest_sha256=sa.digest(candidate/'candidate.json'),m2_manifest_sha256=sa.digest(FROZEN/'candidate.json'),
        code_sha256=sa.digest(Path(__file__)),limits='Reused development cohorts,not independent confirmation.'))
    cells=[c for ds in ('XJTU','MATR','Tongji') for c in sa.load_cells(ds)]
    rows=[];boundary=0.
    with threadpool_limits(limits=1):
        for s in selections:
            branch=ConditionalGeometry(joblib.load(s['model']['path']),s['mode'])
            _,_,test=sa.split(cells,s['fold'])
            for c in test:
                tail=Path('folds')/s['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(candidate/tail) as z:y=z['y'];pp={g:z[g] for g in PARENTS}
                assert np.array_equal(y,c.y[sa.K:])
                q=branch.predict(sa.inference_view(c));yy=c.y.copy();yy[sa.K:]=999;n=min(len(c.x),sa.K+3)
                boundary=max(boundary,float(np.max(abs(q-branch.predict(replace(c,y=yy))))),
                    float(np.max(abs(q[:n-sa.K]-branch.predict(sa.prefix(c,n))))))
                gain=s['selected']['gain']
                for g in PARENTS:pp[g+'4']=pp[g]+gain*(q-pp[g])
                pp['branch_only']=q
                path=out/tail;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,**pp)
                for g,pred in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,pred)))
            print(s['fold'],'all16groups and direct branch scored',flush=True)
    assert len(rows)==6205 and boundary<1e-8
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    failures={a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=look[ds,a][k]-look[ds,b][k]) for ds in ('XJTU','MATR','Tongji') for k in ('mae','rmse','p95_ae') if look[ds,a][k]>=look[ds,b][k]] for a,b in addition_edges()}
    gates={k:not v for k,v in failures.items()}
    error=max(abs(look[r['dataset'],r['group']][k]-r[k]) for r in parent['table'] for k in ('mae','rmse','p95_ae'))
    assert error<1e-8
    assert all(sa.digest(candidate/r['artifact'])==r['sha256'] for r in parent['manifest'])
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in old['manifest'])
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,failures=failures,
        all32_edges_pass=all(gates.values()),rows=len(rows),max_boundary_error=boundary,parent_metric_error=error,
        limits='No independent score audit,mechanism controls,paired intervals or risk stability yet. All point comparisons on repeatedly exposed development data.'))
    print(json.dumps(dict(all32_edges_pass=all(gates.values()),failed=[k for k,v in gates.items() if not v])),flush=True)


if __name__=='__main__':main()
