"""Independent saved-prediction audit and same-configuration representation control."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,error_metrics,aggregate,dump_csv


def independent_metrics(y,p):
    e=np.abs(np.asarray(y,dtype=float)-np.asarray(p,dtype=float))*100
    return dict(mae=float(e.mean()),rmse=float(np.sqrt(np.mean(e**2))),p95_ae=float(np.quantile(e,.95)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=HERE.parent/'results';candidate=root/'m3_waveform_candidate_v1';stability=root/'m3_waveform_stability_v2'
    obj=json.loads((candidate/'candidate.json').read_text());summary=json.loads((stability/'summary.json').read_text())
    sa.write_json(out/'protocol.json',dict(candidate_sha256=sa.digest(candidate/'candidate.json'),
        stability_sha256=sa.digest(stability/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        control='Replace paired PCA8 by absolute PCA8, holding fold kernel, support mode, rank and gain fixed. No query-driven selection.',
        limit='Mechanism diagnostic after development selection; not independent confirmation.'))
    assert all(sa.digest(candidate/r['artifact'])==r['sha256'] for r in obj['manifest'])
    old=json.loads((FROZEN/'candidate.json').read_text())
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in old['manifest'])
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)]
    saved=list(csv.DictReader((stability/'cells.csv').open()))
    lookup={(int(r['seed']),r['cell_id'],r['group']):r for r in saved}
    assert len(saved)==len(lookup)==29200
    verified=[];rows=[];err=0.;affine_error=0.;point_error=0.
    with threadpool_limits(limits=1):
        for fold in sorted({r['fold'] for r in old['manifest']}):
            name=fold.replace(':','__');_,_,test=sa.split(cells,fold)
            sel=json.loads((candidate/'folds'/name/'selection.json').read_text())
            s=sel['original_validation_selection'];gain=s['weight']*sel['global_step']
            src=root/'m3_waveform_screen_v1/folds'/name
            pair=joblib.load(src/(s['key']+'.joblib'))
            control=joblib.load(src/(s['key'].replace('wave_pair_pca8','wave_absolute_pca8')+'.joblib'))
            assert set(pair.source_keys)==set(control.source_keys)=={tuple(k) for k in sel['source_keys']}
            for c in test:
                assert sa.digest(sa.ROOT/c.path)==obj['data_hashes'][c.id]['sha256']
                q=pair.predict(sa.inference_view(c));a=control.predict(sa.inference_view(c))
                with np.load(candidate/'folds'/name/'predictions'/(c.id+'.npz')) as z:
                    y=z['y'];assert np.array_equal(y,c.y[sa.K:])
                    for g in ['B','B1','B2','B12']:
                        p=z[g]+gain*(a-z[g]);actual=z[g]+gain*(q-z[g])
                        point_error=max(point_error,float(np.max(abs(actual-z[g+'3']))))
                        for tag,pred in [('paired',z[g+'3']),('absolute_same_config',p)]:
                            rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g+'3',representation=tag,**error_metrics(y,pred)))
                for seed in range(101,111):
                    with np.load(stability/'folds'/name/str(seed)/(c.id+'.npz')) as z:
                        assert np.array_equal(z['y'],y)
                        for g in ['B','B1','B2','B12']:
                            affine_error=max(affine_error,float(np.max(abs(z[g]+gain*(q-z[g])-z[g+'3']))))
                        for g in ['B','B1','B2','B12','B3','B13','B23','B123']:
                            m=independent_metrics(y,z[g]);r=lookup[seed,c.id,g]
                            err=max(err,max(abs(m[k]-float(r[k])) for k in m))
                            verified.append(dict(seed=seed,dataset=c.dataset,domain=c.domain,group=g,**m))
            print(fold,'saved predictions audited and same-config control scored',flush=True)
    assert max(err,affine_error,point_error)<1e-8
    # Recompute domain-equal means directly, without the production aggregator.
    table={}
    for seed in range(101,111):
        for ds in ['XJTU','MATR','Tongji']:
            for g in ['B','B1','B2','B12','B3','B13','B23','B123']:
                rr=[r for r in verified if r['seed']==seed and r['dataset']==ds and r['group']==g]
                domains=sorted({r['domain'] for r in rr})
                table[seed,ds,g]={m:float(np.mean([np.mean([r[m] for r in rr if r['domain']==d]) for d in domains])) for m in ['mae','rmse','p95_ae']}
    for r in summary['table']:
        assert all(abs(r[m]-table[r['seed'],r['dataset'],r['group']][m])<1e-8 for m in ['mae','rmse','p95_ae'])
    for row in summary['gates']:
        for edge,passed in row['passed'].items():
            a,b=edge.split('<');actual=all(table[row['seed'],ds,a][m]<table[row['seed'],ds,b][m] for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae'])
            assert actual==passed
    controls=aggregate(rows,['dataset','group','representation'])
    dump_csv(out/'control_cells.csv',rows);dump_csv(out/'control_comparison.csv',controls)
    sa.write_json(out/'audit.json',dict(status='PASS',audited_rows=len(verified),metric_error=err,
        affine_replay_error=affine_error,point_replay_error=point_error,stability_pass_counts=summary['pass_counts'],control_table=controls,
        limits=['Same configuration is selected for the paired family, not independently optimized for the absolute control.',
                'Separately validation-selected family controls are available in m3_waveform_shrink_v1.',
                'PCA itself is not a novel contribution; development selection remains disclosed.']))
    print('PASS independent audit; matched representation controls saved',flush=True)


if __name__=='__main__':main()
