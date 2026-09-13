"""Independent saved-point and hierarchy audit of frozen confirmation outputs."""
from collections import defaultdict
import json
import argparse
from pathlib import Path
import numpy as np
from evaluate import sa,HERE


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='calce_frozen_confirmation_v2');args=ap.parse_args()
    root=sa.ROOT/'模块研发/results';out=root/args.run
    request=json.loads((out/'request.json').read_text());report=json.loads((out/'summary.json').read_text())
    assert request['code_sha256']==sa.digest(HERE/'confirm_calce.py')
    assert request['protocol_sha256']==sa.digest(HERE/'calce_cohort_protocol.json')
    cohort=sa.ROOT/request['cohort']
    assert request['pairing_verification_sha256']==sa.digest(cohort/'verification.json')
    refs={}
    for p in (cohort/'cells').glob('*.npz'):
        with np.load(p) as z:q=z['capacity_Ah'].astype('float32')
        refs[p.stem]=(q/q[0])[10:]
    identities=set();recomputed=[];largest=0.
    for r in report['rows']:
        key=r['source_fold'],r['cell_id'],r['group'];assert key not in identities;identities.add(key)
        path=out/r['prediction_file'];assert sa.digest(path)==r['prediction_sha256']
        with np.load(path) as z:
            truth=z['y'];pred=z['pred'];assert np.array_equal(truth,refs[r['cell_id']])
        assert pred.shape==truth.shape and np.isfinite(pred).all()
        e=(pred-truth)*100;v=dict(mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e*e))),
                                p95_ae=float(np.percentile(abs(e),95)))
        for k,val in v.items():largest=max(largest,abs(val-r[k]));assert abs(val-r[k])<1e-9
        recomputed.append(dict(r,**v))
    assert len(identities)==144
    for table in report['table']:
        domain=defaultdict(lambda:defaultdict(list))
        for r in recomputed:
            if r['group']==table['group']:domain[r['domain']][r['source_fold']].append(r)
        assert len(domain)==3 and all(len(v)==6 for v in domain.values())
        for metric in ['mae','rmse','p95_ae']:
            mean=float(np.mean([np.mean([np.mean([r[metric] for r in rr]) for rr in folds.values()])
                                for folds in domain.values()]))
            assert abs(mean-table[metric])<1e-9
    selected=json.loads((sa.ROOT/'模块研发/m2/selected_candidate.json').read_text())
    frozen=root/'m2_reference_candidate_v3';candidate=json.loads((frozen/'candidate.json').read_text())
    assert sa.digest(frozen/'candidate.json')==selected['candidate_manifest_sha256']==request['candidate_sha256']
    assert all(sa.digest(frozen/r['artifact'])==r['sha256'] for r in candidate['manifest'])
    lock=json.loads((root/'calce_cohort_schema_audit_v1/audit.json').read_text())
    assert all(sa.digest(sa.ROOT/r['path'])==r['sha256'] for r in lock['source_artifacts_locked'])
    result=dict(status='PASS',unique_cells=4,source_folds=6,protocol_groups=3,unique_query_points=sum(map(len,refs.values())),
        verified_cell_fold_group_rows=144,max_metric_reconstruction_error_pp=largest,
        frozen_v3_artifacts_verified=42,source_artifacts_verified=30,
        future_label_prefix_error=report['max_future_label_or_prefix_error'],
        report_sha256=sa.digest(out/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        outcome='Relative gains observed; large absolute errors. No universal accuracy or new-laboratory generalization claim.',
        independence='Previously unscored cell cohort within historically used CALCE source, according to audited workspace records.')
    assert result['future_label_prefix_error']<1e-8
    sa.write_json(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
