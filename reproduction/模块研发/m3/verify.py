"""Rebuild all eight-group metrics and test branch replay independently."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa, P, FROZEN, GROUPS, aggregate, error_metrics
from summarize_strict import paired_stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--residual',action='store_true');args=ap.parse_args()
    root=Path(args.out);summary=json.loads((root/'summary.json').read_text())
    candidate=json.loads((FROZEN/'candidate.json').read_text())
    cells=[c for ds in P['datasets'] for c in sa.load_cells(ds)]
    rows=[];replay=0.;counts=[]
    with threadpool_limits(limits=1):
        for fold in sorted({r['fold'] for r in candidate['manifest']}):
            train,val,test=sa.split(cells,fold); dest=root/'folds'/fold.replace(':','__')
            selection=json.loads((dest/'selection.json').read_text())
            assert set(selection['validation_cells'])=={c.id for c in val}
            expected={tuple(k) for k in selection['source_keys']}; counts.append(len(expected))
            assert len(expected)<=1000 and {k[0] for k in expected}=={c.id for c in train}
            models={key:joblib.load(dest/(key+'.joblib')) for key in {s['key'] for s in selection['choices'].values()}}
            training_expected={k for k in expected if k[1]>=sa.K} if args.residual else expected
            assert all(set(model.source_keys)==training_expected for model in models.values())
            for c in test:
                with np.load(dest/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:])
                    for group,s in selection['choices'].items():
                        if args.residual:
                            clip=json.loads((root/'run_protocol.json').read_text())['protocol']['clip']
                            r=models[s['key']].predict(sa.inference_view(c),z[group])
                            p=z[group]+s['weight']*np.clip(r,-clip,clip)
                        else:
                            p=(1-s['weight'])*z[group]+s['weight']*models[s['key']].predict(sa.inference_view(c))
                        replay=max(replay,float(np.max(abs(p-z[group+'3']))))
                    for group in [*GROUPS,*[g+'3' for g in GROUPS]]:
                        rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=group,**error_metrics(z['y'],z[group])))
    assert replay<1e-9 and len(rows)==365*8
    table=aggregate(rows,['dataset','group']);lookup={(r['dataset'],r['group']):r for r in table}
    metric_error=max(abs(lookup[r['dataset'],r['group']][m]-r[m]) for r in summary['table'] for m in ['mae','rmse','p95_ae'])
    assert metric_error<1e-10
    # Original four-group metrics must match the frozen candidate, not a renamed replacement.
    frozen_table=candidate['table']
    reverse={v:k for k,v in GROUPS.items()}
    parent_metric_error=max(abs(lookup[r['dataset'],reverse[r['group']]][m]-100*r[m])
        for r in frozen_table for m in ['mae','rmse','p95_ae'])
    assert parent_metric_error<1e-8
    sa.write_json(root/'reconstructed_table.json',table)
    pairs=[]
    for a,b in [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3')]:
        for ds in P['datasets']:
            aa={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==a}
            bb={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==b}
            for m in ['mae','rmse','p95_ae']:
                pairs.append(dict(comparison=a+'-'+b,dataset=ds,metric=m,**paired_stats(
                    [(aa[c][m]-bb[c][m])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in candidate['manifest'])
    sa.write_json(root/'verification.json',dict(status='PASS_BRANCH_REPLAY_AND_METRICS',rows=len(rows),
        max_replay_error=replay,max_metric_error=metric_error,max_frozen_parent_metric_error=parent_metric_error,
        source_budget_range=[min(counts),max(counts)],
        frozen_artifact_hashes=42,paired_comparisons=pairs,
        limits=['Not an independent final test; intervals are exploratory after development selection.',
                'Full frozen-parent cache replay remains a separate audit before adoption.']))
    print('PASS',len(rows),'rows',replay,metric_error,flush=True)


if __name__=='__main__':main()
