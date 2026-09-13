"""Independent M4 score replay, domain aggregation, all32 gates and paired intervals."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from score import PARENTS,addition_edges
from summarize_strict import paired_stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root)
    summary=json.loads((root/'summary.json').read_text());frozen=json.loads((root/'frozen_selections.json').read_text())
    candidate=HERE.parent/'results/m3_waveform_candidate_v1'
    assert sa.digest(candidate/'candidate.json')==frozen['candidate_sha256']
    rows=list(csv.DictReader((root/'cells.csv').open()));saved={(r['cell_id'],r['group']):r for r in rows}
    assert len(rows)==len(saved)==8760
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];recomputed=[];error=0.;replay=0.
    with threadpool_limits(limits=1):
        for selection in frozen['selections']:
            fold=selection['fold'];_,_,test=sa.split(cells,fold);models={};controls={};gain=selection['selected']['gain']
            for g in PARENTS:
                for label,dest in [('models',models),('controls',controls)]:
                    spec=selection[label][g];path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];dest[g]=joblib.load(path)
            for c in test:
                tail=Path('folds')/fold.replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(root/tail) as z,np.load(candidate/tail) as old:
                    assert np.array_equal(z['y'],c.y[sa.K:]) and np.array_equal(z['y'],old['y'])
                    expected={g:old[g] for g in PARENTS}
                    for g in PARENTS:
                        expected[g+'4']=old[g]+gain*models[g].predict(sa.inference_view(c),old[g])
                        expected[g+'4_constant']=old[g]+gain*controls[g].predict(sa.inference_view(c),old[g])
                    assert set(z.files)=={'y'}|set(expected)
                    for g,p in expected.items():
                        replay=max(replay,float(np.max(abs(p-z[g]))))
                        e=(np.asarray(z[g],dtype=float)-np.asarray(z['y'],dtype=float))*100;a=abs(e)
                        metrics=dict(mae=float(a.sum()/len(a)),rmse=float(np.sqrt(np.dot(e,e)/len(e))),p95_ae=float(np.percentile(a,95)))
                        error=max(error,max(abs(metrics[k]-float(saved[c.id,g][k])) for k in metrics))
                        recomputed.append(dict(cell_id=c.id,dataset=c.dataset,domain=c.domain,group=g,**metrics))
            print(fold,'24 groups independently replayed and metrics recalculated',flush=True)
    assert max(error,replay)<1e-8 and len(recomputed)==8760
    table={}
    for r in summary['table']:
        rr=[v for v in recomputed if v['dataset']==r['dataset'] and v['group']==r['group']];domains={v['domain'] for v in rr}
        mm={k:float(np.mean([np.mean([v[k] for v in rr if v['domain']==d]) for d in domains])) for k in ['mae','rmse','p95_ae']}
        assert all(abs(r[k]-mm[k])<1e-8 for k in mm);table[r['dataset'],r['group']]=mm
    pairs=[]
    for a,b in addition_edges():
        passed=all(table[ds,a][k]<table[ds,b][k] for ds in ['XJTU','MATR','Tongji'] for k in ['mae','rmse','p95_ae'])
        assert summary['gates'][a+'<'+b]==passed
        for ds in ['XJTU','MATR','Tongji']:
            aa={v['cell_id']:v for v in recomputed if v['dataset']==ds and v['group']==a}
            bb={v['cell_id']:v for v in recomputed if v['dataset']==ds and v['group']==b}
            assert set(aa)==set(bb)
            for k in ['mae','rmse','p95_ae']:
                pairs.append(dict(comparison=a+'-'+b,dataset=ds,metric=k,**paired_stats([(aa[c][k]-bb[c][k])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    assert len(pairs)==288
    sa.write_json(root/'verification.json',dict(status='PASS',replayed_rows=len(recomputed),prediction_error=replay,metric_error=error,
        all32_gates_reproduced=True,paired_comparisons=pairs,summary_sha256=sa.digest(root/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Development paired intervals without selection/multiplicity correction; not independent confirmation or guaranteed coverage.'))
    print('PASS score audit and288 paired comparisons; passing audit does not imply passing performance',flush=True)


if __name__=='__main__':main()
