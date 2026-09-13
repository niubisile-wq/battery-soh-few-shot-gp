"""Replay chosen coverage models and reconstruct gated/ungated output tables."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,FROZEN,GROUPS,error_metrics,aggregate
from coverage_gate import coverage,trust


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root)
    summary=json.loads((root/'summary.json').read_text());manifest=json.loads((FROZEN/'candidate.json').read_text())
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[];error=0.;branch_error=0.;trust_error=0.
    with threadpool_limits(limits=1):
        for dest in sorted((root/'folds').iterdir()):
            fold=dest.name.replace('__',':',1);_,val,test=sa.split(cells,fold)
            selection=json.loads((dest/'selection.json').read_text());s=selection['chosen']
            assert set(selection['validation_cells'])=={c.id for c in val}
            best=min(selection['trials'],key=lambda r:(r['validation_mae'],r['weight'],r['path'],r['kind'],r['power']))
            assert best==s
            assert sa.digest(s['path'])==summary['source_hashes'][s['path']]
            model=joblib.load(s['path']);assert set(model.source_keys)=={tuple(k) for k in selection['source_keys']}
            for c in test:
                p=model.predict(sa.inference_view(c));q,r=coverage(model,sa.inference_view(c));t=trust(q,r,s['kind'],s['power']);w=s['weight']
                with np.load(dest/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:])
                    branch_error=max(branch_error,float(np.max(abs(p-z['branch']))));trust_error=max(trust_error,float(np.max(abs(t-z['trust']))))
                    for g in GROUPS:
                        for name,expected in [(g,z[g]),(g+'3',z[g]+w*t*(p-z[g])),(g+'3_ungated',z[g]+w*(p-z[g]))]:
                            error=max(error,float(np.max(abs(expected-z[name]))))
                            rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=name,**error_metrics(z['y'],z[name])))
            print(fold,'coverage replay verified',flush=True)
    assert len(rows)==4380 and max(error,branch_error,trust_error)<1e-8
    table=aggregate(rows,['dataset','group']);lookup={(r['dataset'],r['group']):r for r in table}
    metric_error=max(abs(lookup[r['dataset'],r['group']][m]-r[m]) for r in summary['table'] for m in ['mae','rmse','p95_ae'])
    reverse={v:k for k,v in GROUPS.items()}
    parent_error=max(abs(lookup[r['dataset'],reverse[r['group']]][m]-100*r[m]) for r in manifest['table'] for m in ['mae','rmse','p95_ae'])
    assert max(metric_error,parent_error)<1e-8 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    sa.write_json(root/'verification.json',dict(status='PASS',rows=len(rows),branch_replay_error=branch_error,trust_replay_error=trust_error,
        output_replay_error=error,max_metric_error=metric_error,max_frozen_parent_metric_error=parent_error,
        limits=['Frozen parent aggregate/cache check, not another full parent-model replay.', 'Development screening, not independent confirmation.']))
    print('PASS',len(rows),error,flush=True)


if __name__=='__main__':main()
