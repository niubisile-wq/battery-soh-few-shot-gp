"""Refit selected source reliability gates and replay complete stored ablations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,SCREEN,GROUPS,error_metrics,aggregate
from reliability import ReliabilityGate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root)
    summary=json.loads((root/'summary.json').read_text());run=json.loads((root/'run_protocol.json').read_text())
    for name,h in run['hashes'].items():assert sa.digest(HERE/name)==h
    manifest=json.loads((FROZEN/'candidate.json').read_text());source_root=HERE.parent/'results'/run['protocol']['source_oof_run']
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[];replay=0.;coef_error=0.;negative_vars=[]
    with threadpool_limits(limits=1):
        for dest in sorted((root/'folds').iterdir()):
            fold=dest.name.replace('__',':',1);tr,val,test=sa.split(cells,fold)
            ss=json.loads((dest/'selection.json').read_text());s=ss['chosen'];assert set(ss['validation_cells'])=={c.id for c in val}
            best=min(ss['trials'],key=lambda r:(r['validation_mae'],r['gain'],r['key']))
            assert all(best[k]==s[k] for k in best)
            assert sa.digest(s['branch_path'])==s['branch_sha256'];branch=joblib.load(s['branch_path'])
            allowed=defaultdict(list)
            for cid,j in ss['source_keys']:allowed[cid].append(j)
            assert set(allowed)=={c.id for c in tr};assert len(ss['source_keys'])<=1000
            src=joblib.load(SCREEN/dest.name/'seed_0/source_lodo_episodes.joblib');oof=joblib.load(source_root/'folds'/dest.name/'source_oof_predictions.joblib')
            models={}
            for family in ['B','B1']:
                m=joblib.load(dest/(family+'__'+s['key']+'.joblib'));models[family]=m
                assert set(m.source_keys)=={(cid,j) for cid,idx in allowed.items() for j in idx if j>=sa.K}
                assert np.isfinite(m.scaler.scale_).all() and np.isfinite(m.regressor.coef_).all()
                negative_vars.extend(m.scaler.var_[m.scaler.var_<0].tolist())
                refit=ReliabilityGate(m.view,m.alpha).fit(src[GROUPS[family]],oof,[sa.source_view(c,allowed[c.id]) for c in tr],allowed)
                coef_error=max(coef_error,float(np.max(abs(refit.regressor.coef_-m.regressor.coef_))))
                np.testing.assert_allclose(refit.scaler.scale_,m.scaler.scale_,rtol=0,atol=1e-10)
                np.testing.assert_allclose(refit.regressor.intercept_,m.regressor.intercept_,rtol=0,atol=1e-10)
            for c in test:
                b=branch.predict(sa.inference_view(c))
                with np.load(dest/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:]);replay=max(replay,float(np.max(abs(b-z['branch']))))
                    for g in GROUPS:
                        m=models['B1' if '1' in g else 'B'];w=m.predict(sa.inference_view(c),z[g],b)
                        replay=max(replay,float(np.max(abs(w-z[g+'_weight']))))
                        pp={g:z[g],g+'3':z[g]+s['gain']*w*(b-z[g]),g+'3_constant':z[g]+s['gain']*m.constant*(b-z[g])}
                        for name,p in pp.items():
                            replay=max(replay,float(np.max(abs(p-z[name]))));rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=name,**error_metrics(z['y'],z[name])))
            print(fold,'gate refit and prediction replay verified',flush=True)
    assert len(rows)==4380 and max(replay,coef_error)<1e-8
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    metric_error=max(abs(look[r['dataset'],r['group']][m]-r[m]) for r in summary['table'] for m in ['mae','rmse','p95_ae'])
    reverse={v:k for k,v in GROUPS.items()};parent_error=max(abs(look[r['dataset'],reverse[r['group']]][m]-100*r[m]) for r in manifest['table'] for m in ['mae','rmse','p95_ae'])
    assert max(metric_error,parent_error)<1e-8 and all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    sa.write_json(root/'verification.json',dict(status='PASS',rows=len(rows),selected_gate_refits=42,max_coefficient_error=coef_error,max_prediction_error=replay,
        max_metric_error=metric_error,max_parent_metric_error=parent_error,minimum_weighted_variance=min(negative_vars,default=0),
        limits=['Tiny negative weighted variance from roundoff; scaler constant handling yields finite scales.',
                'Old parent aggregate/cache check, not a new full old-parent replay.', 'Development, not independent confirmation.']))
    print('PASS',len(rows),replay,coef_error,flush=True)


if __name__=='__main__':main()
