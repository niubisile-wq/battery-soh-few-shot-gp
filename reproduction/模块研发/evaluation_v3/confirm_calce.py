"""One frozen confirmation on the audited CALCE cell cohort. No fitting/selection."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from evaluate import sa,HERE,error_metrics,aggregate,dump_csv
from data import Cell


def predict_groups(c,full,independent,pls,mode):
    masked=sa.inference_view(c)
    physical=full.physical.predict(masked,full.physical_mode)
    ordinary=pls.predict(masked,mode)
    return dict(GPR=independent.parent.predict(masked,independent.parent_mode),
        M1=full.parent.predict(masked,full.parent_mode),GPR_M2=independent.predict(c),
        Full_v3=full.predict(c),ordinary_PLS=ordinary,ordinary_PLS_half_physical=(ordinary+physical)/2)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--cohort',default='calce_cohort_extraction_v3')
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=sa.ROOT/'模块研发/results';cohort=root/args.cohort
    verification=json.loads((cohort/'verification.json').read_text())
    assert verification['status']=='PASS_FOR_FROZEN_CELL_COHORT_CONFIRMATION'
    protocol_path=HERE/'calce_cohort_protocol.json';protocol=json.loads(protocol_path.read_text())
    assert verification['protocol_sha256']==sa.digest(protocol_path)
    lock=json.loads((root/'calce_cohort_schema_audit_v1/audit.json').read_text())
    assert all(sa.digest(sa.ROOT/r['path'])==r['sha256'] for r in lock['source_artifacts_locked'])
    frozen=root/'m2_reference_candidate_v3';candidate=json.loads((frozen/'candidate.json').read_text())
    assert all(sa.digest(frozen/r['artifact'])==r['sha256'] for r in candidate['manifest'])
    request=dict(status='RUNNING_FROZEN_NO_SELECTION',protocol_sha256=sa.digest(protocol_path),
        cohort=str(cohort.relative_to(sa.ROOT)),
        code_sha256=sa.digest(Path(__file__)),pairing_verification_sha256=sa.digest(cohort/'verification.json'),
        source_lock_sha256=sa.digest(root/'calce_cohort_schema_audit_v1/audit.json'),
        candidate_sha256=sa.digest(frozen/'candidate.json'),expected_cell_fold_group_rows=4*6*6)
    sa.write_json(out/'request.json',request)
    cells=[]
    for spec in verification['cells']:
        cid=spec['cell_id'];p=cohort/'cells'/(cid+'.npz');assert sa.digest(p)==spec['cache_sha256']
        with np.load(p) as z:
            q=z['capacity_Ah'].astype('float32')
            cells.append(Cell(cid,'CALCE_unscored_cells',str(z['domain']),z['x'].astype('float32'),
                q/q[0],z['cycle_number'].astype('float32'),float(q[0]),float(z['nominal_capacity_Ah']),str(p.relative_to(sa.ROOT))))
    folds=sorted({r['fold'] for r in candidate['manifest'] if r['fold'].startswith('XJTU:')})
    assert len(folds)==6
    rows=[];boundary=0.;start=time.monotonic()
    with threadpool_limits(limits=1):
        for fold in folds:
            name=fold.replace(':','__')
            full=joblib.load(frozen/'folds'/name/'Base+M1+M2/adapter.joblib')
            independent=joblib.load(frozen/'folds'/name/'Base+M2/adapter.joblib')
            job=root/'m1_v1/screen_v1/jobs'/('P04_pls_metric__'+name)
            pls=joblib.load(job/'model.joblib');mode=json.loads((job/'tuning.json').read_text())['chosen']['mode']
            for c in cells:
                predictions=predict_groups(c,full,independent,pls,mode)
                y=c.y.copy();y[sa.K:]=123.
                alt=predict_groups(replace(c,y=y),full,independent,pls,mode)
                prefix=predict_groups(sa.prefix(c,sa.K+3),full,independent,pls,mode)
                assert set(predictions)==set(protocol['groups'])
                for group,p in predictions.items():
                    delta=max(float(np.max(abs(alt[group]-p))),float(np.max(abs(prefix[group]-p[:3]))))
                    boundary=max(boundary,delta);assert delta<1e-8
                    path=out/'predictions'/name/group/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,y=c.y[sa.K:],pred=p)
                    rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,source_fold=fold,group=group,
                        prediction_file=str(path.relative_to(out)),prediction_sha256=sa.digest(path),**error_metrics(c.y[sa.K:],p)))
                print(fold,c.id,'frozen predictions and boundaries complete',flush=True)
    assert len(rows)==144
    table=aggregate(rows,['group'])
    for r in table:
        r['cell_fold_rows']=r.pop('cells');r['unique_cells']=4;r['source_folds']=6
        r['evaluated_query_predictions']=r.pop('points');r['unique_query_points']=sum(len(c.y)-10 for c in cells)
    cell_table=aggregate(rows,['cell_id','group']);domain_table=aggregate(rows,['domain','group'])
    comparisons=[];rng=np.random.default_rng(20260908)
    for reference in [g for g in protocol['groups'] if g!='Full_v3']:
        for metric in ['mae','rmse','p95_ae']:
            delta=[]
            for domain in sorted({c.domain for c in cells}):
                a=next(r for r in domain_table if r['domain']==domain and r['group']=='Full_v3')
                b=next(r for r in domain_table if r['domain']==domain and r['group']==reference)
                delta.append(a[metric]-b[metric])
            draws=np.mean(np.asarray(delta)[rng.integers(0,len(delta),(5000,len(delta)))],axis=1)
            comparisons.append(dict(reference=reference,metric=metric,full_minus_reference_pp=float(np.mean(delta)),
                protocol_deltas_pp=delta,protocol_bootstrap_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
                limits='Only three observed protocol groups; overlapping source folds are not independent samples.'))
    assert all(sa.digest(sa.ROOT/r['path'])==r['sha256'] for r in lock['source_artifacts_locked'])
    assert all(sa.digest(frozen/r['artifact'])==r['sha256'] for r in candidate['manifest'])
    report=dict(status='FROZEN_CONFIRMATION_SCORED_REQUIRES_OUTPUT_AUDIT',table=table,cells=cell_table,
        domains=domain_table,paired_comparisons=comparisons,rows=rows,elapsed_seconds=time.monotonic()-start,
        max_future_label_or_prefix_error=boundary,selection_performed=False,new_training_performed=False,
        limits=verification['limits']+['No source-fold prediction ensemble; budgets apply per source-fold model.',
            'This is a final small-cohort check, not proof of universal generalization or journal acceptance.',
            'Report all outcomes, including degradation; do not use these scores to repair or select M3.'])
    sa.write_json(out/'summary.json',report);dump_csv(out/'comparison.csv',table);dump_csv(out/'cell_fold_results.csv',rows)
    request.update(status='COMPLETE_SCORED_NOT_YET_OUTPUT_AUDITED',scored_rows=len(rows));sa.write_json(out/'request.json',request)
    print(json.dumps(table,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
