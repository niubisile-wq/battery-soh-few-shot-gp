"""Reconstruct reported control/probe metrics from cached point predictions."""
import csv
import json
from pathlib import Path
import numpy as np
from evaluate import sa, error_metrics, aggregate, HERE
from probe_m3 import correction


def main():
    root=sa.ROOT/'模块研发/results'
    diag=json.loads((root/'frozen_v3_evaluation_v2/summary.json').read_text())
    probe=json.loads((root/'frozen_v3_m3_probe_v1/summary.json').read_text())
    frozen=root/'m2_reference_candidate_v3'
    candidate=json.loads((frozen/'candidate.json').read_text())
    assert all(sa.digest(frozen/r['artifact'])==r['sha256'] for r in candidate['manifest'])
    assert diag['protocol_sha256']==sa.digest(HERE/'protocol.json')
    assert probe['protocol_sha256']==sa.digest(HERE/'m3_protocol.json')
    assert diag['code_sha256']==sa.digest(HERE/'evaluate.py')
    assert probe['code_sha256']==sa.digest(HERE/'probe_m3.py')
    tables={}
    for name in ['frozen_v3_evaluation_v2','frozen_v3_m3_probe_v1']:
        with (root/name/'cells.csv').open() as f: rr=list(csv.DictReader(f))
        tables[name]={(r['cell_id'],r['group']):r for r in rr}
        assert len(rr)==len(tables[name])
    weights={(r['fold'],r['parent']):r['weight'] for r in diag['selections']}
    strengths={(r['fold'],r['direction']):r['strength'] for r in probe['selections']}
    maximum=0.;count=0
    for ds in ['XJTU','MATR','Tongji']:
        for c in sa.load_cells(ds):
            fold=f'{ds}:{c.domain}';name=fold.replace(':','__')
            job=root/'m2_reference_bagged_screen_v1/folds'/name/'seed_0'
            with np.load(job/'Base+M1+M2'/(c.id+'.npz')) as z:
                y=z['y'];preds={'M1':z['parent'],'Full_v3':z[candidate['variant']],'physical':z['physical_control']}
            with np.load(job/'Base+M2'/(c.id+'.npz')) as z:
                preds.update(GPR=z['parent'],GPR_M2=z[candidate['variant']])
            with np.load(root/'m1_v1/screen_v1/jobs'/('P04_pls_metric__'+name)/'predictions'/(Path(c.id).stem+'.npz')) as z:
                preds['ordinary_PLS']=z['pred']
            for parent in ['M1','ordinary_PLS']:
                w=weights[fold,parent]
                preds[parent+'_val_physical']=(1-w)*preds[parent]+w*preds['physical']
                preds[parent+'_half_physical']=(preds[parent]+preds['physical'])/2
            p3={'Full_v3':preds['Full_v3']}
            for direction in ['degradation_gain','charge_anchor']:
                p3[direction]=correction(c,preds['Full_v3'],direction,strengths[fold,direction])
            for run,pp in [('frozen_v3_evaluation_v2',preds),('frozen_v3_m3_probe_v1',p3)]:
                for group,p in pp.items():
                    actual=error_metrics(y,p);old=tables[run][c.id,group]
                    delta=max(abs(actual[k]-float(old[k])) for k in actual)
                    maximum=max(maximum,delta);assert delta < 1e-7
                    count+=1
    assert count==365*13
    assert not any(r['pass_point_gate'] for r in probe['gates'].values())
    report=dict(status='PASS',reconstructed_cell_group_rows=count,max_metric_difference_pp=maximum,
                frozen_artifacts_verified=len(candidate['manifest']),code_sha256=sa.digest(Path(__file__)),
                scope='Cached-point metric reconstruction for all 10 controls and 3 probe groups; M3 inference replay/boundaries separately in probe summary',
                independent_final_validation='NOT_DONE: historical exposure audit and user data-source decision pending')
    sa.write_json(root/'frozen_v3_evaluation_v2/verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
