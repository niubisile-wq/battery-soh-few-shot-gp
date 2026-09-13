"""Independent direct-branch replay, equal-domain metrics and paired ablations."""
import csv
import argparse
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from conditional_geometry import ConditionalGeometry
from score import PARENTS,addition_edges
from summarize_strict import paired_stats


def main():
    ap=argparse.ArgumentParser();variants=ap.add_mutually_exclusive_group();variants.add_argument('--support-gate',action='store_true');variants.add_argument('--structured',action='store_true');variants.add_argument('--registered',action='store_true');variants.add_argument('--registered-gate',action='store_true');args=ap.parse_args()
    variant='registered_gate' if args.registered_gate else 'support_gate' if args.support_gate else 'registered' if args.registered else 'structured' if args.structured else 'multiscale'
    args.support_gate=args.support_gate or args.registered_gate
    results=HERE.parent/'results';root=results/('m4_'+variant+'_screen_v1');parent=results/'m3_waveform_candidate_v1'
    expected_rows=9125 if args.support_gate else 6205
    summary=json.loads((root/'summary.json').read_text());frozen=json.loads((root/'frozen_selections.json').read_text())
    assert sa.digest(parent/'candidate.json')==frozen['parent_sha256' if args.support_gate else 'parent_manifest_sha256']
    with (root/'cells.csv').open() as f:rows=list(csv.DictReader(f))
    saved={(r['cell_id'],r['group']):r for r in rows};assert len(saved)==len(rows)==expected_rows
    cells=[c for ds in ('XJTU','MATR','Tongji') for c in sa.load_cells(ds)]
    recalculated=[];replay=metric_error=0.
    with threadpool_limits(limits=1):
        for s in frozen['selections']:
            if args.support_gate:
                assert sa.digest(Path(s['gate']['path']))==s['gate']['sha256']
                assert sa.digest(Path(s['result']))==s['result_sha256']
                gate=joblib.load(s['gate']['path'])
                model=ConditionalGeometry(gate.model,s['selected']['key'].split('__')[-1])
            else:
                assert sa.digest(Path(s['model']['path']))==s['model']['sha256']
                assert sa.digest(Path(s['branch_root'])/'result.json')==s['result_sha256']
                model=ConditionalGeometry(joblib.load(s['model']['path']),s['mode'])
            _,_,test=sa.split(cells,s['fold'])
            for c in test:
                tail=Path('folds')/s['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(root/tail) as z,np.load(parent/tail) as old:
                    assert np.array_equal(z['y'],old['y']) and np.array_equal(z['y'],c.y[sa.K:])
                    branch=model.predict(sa.inference_view(c));expected={g:old[g] for g in PARENTS}
                    weight=1.
                    if args.support_gate:
                        latent=gate.model.scaler.transform(gate.model.feature_matrix(sa.inference_view(c)))[sa.K:]
                        nearest=cdist(latent,gate.model.gp.X_train_,'sqeuclidean').min(1)
                        scale=gate.scale2*s['selected']['multiplier'];weight=scale/(scale+nearest)
                        replay=max(replay,float(np.max(abs(weight-z['gate']))))
                    for g in PARENTS:
                        amount=s['selected']['gain']*weight
                        expected[g+'4']=(1-amount)*old[g]+amount*branch
                        if args.support_gate:expected[g+'4_ungated']=(1-s['selected']['gain'])*old[g]+s['selected']['gain']*branch
                    expected['branch_only']=branch;assert set(z.files)==({'y','gate'} if args.support_gate else {'y'})|set(expected)
                    for g,p in expected.items():
                        replay=max(replay,float(np.max(abs(p-z[g]))))
                        e=(np.asarray(p,float)-np.asarray(z['y'],float))*100
                        metrics=dict(mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e**2))),p95_ae=float(np.quantile(abs(e),.95)))
                        row=saved[c.id,g];assert row['dataset']==c.dataset and row['domain']==c.domain
                        metric_error=max(metric_error,max(abs(v-float(row[k])) for k,v in metrics.items()))
                        recalculated.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**metrics))
            print(s['fold'],25 if args.support_gate else 17,'outputs independently replayed',flush=True)
    assert len(recalculated)==expected_rows and max(replay,metric_error)<1e-8
    buckets=defaultdict(list)
    for r in recalculated:buckets[r['dataset'],r['group'],r['domain']].append([r[k] for k in ('mae','rmse','p95_ae')])
    means=defaultdict(list)
    for (ds,g,d),values in buckets.items():means[ds,g].append(np.mean(values,axis=0))
    table={k:np.mean(v,axis=0) for k,v in means.items()}
    assert len(table)==len(summary['table'])
    for r in summary['table']:assert np.max(abs(table[r['dataset'],r['group']]-[r[k] for k in ('mae','rmse','p95_ae')]))<1e-8
    pairs=[]
    for a,b in addition_edges():
        passed=all(np.all(table[ds,a]<table[ds,b]) for ds in ('XJTU','MATR','Tongji'))
        assert summary['gates'][a+'<'+b]==passed
        for ds in ('XJTU','MATR','Tongji'):
            aa={r['cell_id']:r for r in recalculated if r['dataset']==ds and r['group']==a}
            bb={r['cell_id']:r for r in recalculated if r['dataset']==ds and r['group']==b}
            assert set(aa)==set(bb)
            for k in ('mae','rmse','p95_ae'):
                pairs.append(dict(comparison=a+'-'+b,dataset=ds,metric=k,**paired_stats([(aa[c][k]-bb[c][k])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    assert len(pairs)==288
    sa.write_json(root/'verification.json',dict(status='PASS',rows=len(recalculated),prediction_error=replay,metric_error=metric_error,
        all32_gates_reproduced=True,paired_comparisons=pairs,summary_sha256=sa.digest(root/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Branch replay from saved models and parent caches; no new all-model source refit. Development intervals without selection/multiplicity correction,not independent confirmation.'))
    print(f'PASS{expected_rows}rows and288paired comparisons; audit pass is not performance pass',flush=True)


if __name__=='__main__':main()
