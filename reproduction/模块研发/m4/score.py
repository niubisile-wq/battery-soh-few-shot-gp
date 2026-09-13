"""Score frozen all-fold M4 selections, all16 groups and constant controls."""
import argparse
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN
from correction import ResidualCorrector
from evaluate import aggregate,error_metrics,dump_csv

PARENTS=['B','B1','B2','B12','B3','B13','B23','B123']


def addition_edges():
    subsets=['']+[''.join(s) for n in range(1,5) for s in combinations('1234',n)]
    return [('B'+''.join(sorted(s+m)),'B'+s) for s in subsets for m in '1234' if m not in s]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    batch=Path(args.batch).resolve();obj=json.loads((batch/'result.json').read_text());assert obj['status']=='VALIDATION_BATCH_COMPLETE'
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=HERE.parent/'results';candidate=root/'m3_waveform_candidate_v1';frozen=json.loads((candidate/'candidate.json').read_text())
    assert {r['fold'] for r in obj['folds']}=={r['fold'] for r in frozen['manifest']}
    # Bind ALL choices before opening outer prediction files.
    selections=[];modelpaths={}
    for entry in obj['folds']:
        correction=Path(entry['correction']);source=Path(entry['source'])
        assert json.loads((source/'verification.json').read_text())['status']=='PASS'
        audit_path=correction/'verification.json'
        if not audit_path.exists():
            subprocess.run([sys.executable,str(HERE/'audit_correction.py'),'--source',str(source),'--correction',str(correction)],check=True)
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(correction/'result.json')
        assert audit['code_sha256']==sa.digest(HERE/'audit_correction.py')
        result=json.loads((correction/'result.json').read_text());sel=result['selected']
        expected=min(result['trials'],key=lambda r:(r['validation_mae'],r['gain'],r['key']));assert sel==expected
        paths={g:correction/'models'/(g+'__'+sel['key']+'.joblib') for g in PARENTS}
        controls={g:correction/'models'/(g+'__constant.joblib') for g in PARENTS}
        modelpaths[entry['fold']]=(paths,controls)
        selections.append(dict(fold=entry['fold'],selected=sel,selection_sha256=sa.digest(correction/'result.json'),
            models={g:dict(path=str(p),sha256=sa.digest(p)) for g,p in paths.items()},
            controls={g:dict(path=str(p),sha256=sa.digest(p)) for g,p in controls.items()}))
    assert len(selections)==21
    sa.write_json(out/'frozen_selections.json',dict(selections=selections,code_sha256=sa.digest(Path(__file__)),
        candidate_sha256=sa.digest(candidate/'candidate.json'),limits='Development screen, not independent confirmation. Constant control uses identical gain.'))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[];boundary=0.
    with threadpool_limits(limits=1):
        for entry in selections:
            fold=entry['fold'];name=fold.replace(':','__');_,val,test=sa.split(cells,fold)
            paths,controlpaths=modelpaths[fold];models={g:joblib.load(p) for g,p in paths.items()};controls={g:joblib.load(p) for g,p in controlpaths.items()}
            gain=entry['selected']['gain'];dest=out/'folds'/name
            for c in test:
                with np.load(candidate/'folds'/name/'predictions'/(c.id+'.npz')) as z:
                    y=z['y'];assert np.array_equal(y,c.y[sa.K:]);pp={g:z[g].copy() for g in PARENTS}
                for g in PARENTS:
                    r=models[g].predict(c,pp[g]);pp[g+'4']=pp[g]+gain*r
                    pp[g+'4_constant']=pp[g]+gain*controls[g].predict(c,pp[g])
                    yy=c.y.copy();yy[sa.K:]=123;n=min(len(c.x),sa.K+3)
                    boundary=max(boundary,float(np.max(abs(r-models[g].predict(replace(c,y=yy),pp[g])))),
                        float(np.max(abs(r[:n-sa.K]-models[g].predict(sa.prefix(c,n),pp[g][:n-sa.K])))))
                assert boundary<1e-8
                path=dest/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,**pp)
                for g,pred in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,pred)))
            print(fold,'16 groups and8 constant controls scored',flush=True)
    assert len(rows)==8760
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    failures={a+'<'+b:[dict(dataset=ds,metric=m,delta_pp=look[ds,a][m]-look[ds,b][m])
        for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae'] if look[ds,a][m]>=look[ds,b][m]] for a,b in addition_edges()}
    gates={k:not v for k,v in failures.items()}
    parenterr=max(abs(look[r['dataset'],r['group']][m]-r[m]) for r in frozen['table'] for m in ['mae','rmse','p95_ae'])
    assert parenterr<1e-8 and all(sa.digest(candidate/r['artifact'])==r['sha256'] for r in frozen['manifest'])
    old=json.loads((FROZEN/'candidate.json').read_text());assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in old['manifest'])
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,failures=failures,
        all32_edges_pass=all(gates.values()),max_boundary_error=boundary,max_frozen_parent_metric_error=parenterr,
        limits=['Per-fold validation selected, extensively reused development data, not final independent test.',
                'Point comparisons do not imply significance; paired uncertainty and risk stability remain pending.']))
    print(json.dumps(dict(all32_edges_pass=all(gates.values()),failed_edges=[k for k,v in gates.items() if not v])),flush=True)


if __name__=='__main__':main()
