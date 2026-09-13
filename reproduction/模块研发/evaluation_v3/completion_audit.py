"""Requirement-level audit of this evaluation, not a model-performance acceptance."""
import json
from pathlib import Path
import subprocess
import sys
from evaluate import sa,HERE


def main():
    root=sa.ROOT;results=root/'模块研发/results';evidence={}
    def read(relative):
        path=root/relative;evidence[relative]=sa.digest(path)
        return json.loads(path.read_text())
    diagnostic=read('模块研发/results/frozen_v3_evaluation_v2/summary.json')
    assert diagnostic['status']=='COMPLETE' and diagnostic['cells']==365 and diagnostic['folds']==21
    assert {r['dataset'] for r in diagnostic['bins']}=={'XJTU','MATR','Tongji'}
    assert all(any(r['group']==g and r['dataset']==ds for r in diagnostic['table'])
               for g in ['ordinary_PLS','ordinary_PLS_half_physical','ordinary_PLS_val_physical','Full_v3']
               for ds in ['XJTU','MATR','Tongji'])
    assert len(diagnostic['selections'])==42
    diagnosis_verify=read('模块研发/results/frozen_v3_evaluation_v2/verification.json')
    assert diagnosis_verify['status']=='PASS' and diagnosis_verify['reconstructed_cell_group_rows']==4745
    probe=read('模块研发/results/frozen_v3_m3_probe_v1/summary.json')
    assert probe['status']=='COMPLETE' and len(probe['gates'])==2
    assert not any(r['pass_point_gate'] for r in probe['gates'].values())
    assert probe['max_query_label_or_prefix_error']<1e-8
    assert probe['protocol_sha256']==sa.digest(HERE/'m3_protocol.json')
    exposure=read('模块研发/results/calce_unused_cohort_audit_v1/audit.json')
    assert exposure['candidate_cells']==14 and exposure['searched_text_files']==7872
    pairing=read('模块研发/results/calce_cohort_extraction_v3/verification.json')
    assert pairing['status']=='PASS_FOR_FROZEN_CELL_COHORT_CONFIRMATION' and pairing['verified_pairs']==3875
    confirmation=read('模块研发/results/calce_frozen_confirmation_v2/verification.json')
    assert confirmation['status']=='PASS' and confirmation['verified_cell_fold_group_rows']==144
    scored=read('模块研发/results/calce_frozen_confirmation_v2/summary.json')
    assert confirmation['report_sha256']==evidence['模块研发/results/calce_frozen_confirmation_v2/summary.json']
    assert not scored['selection_performed'] and not scored['new_training_performed']
    correction=read('模块研发/results/calce_frozen_confirmation_v2/correction_verification.json')
    assert correction['status']=='PASS' and correction['removed_measurements']==4
    assert correction['unchanged_support_cells']==4 and correction['max_retained_prediction_change']==0
    selected=read('模块研发/m2/selected_candidate.json')
    candidate=read('模块研发/results/m2_reference_candidate_v3/candidate.json')
    assert selected['candidate_manifest_sha256']==evidence['模块研发/results/m2_reference_candidate_v3/candidate.json']
    assert len(candidate['manifest'])==42
    for r in candidate['manifest']:
        assert sa.digest(results/'m2_reference_candidate_v3'/r['artifact'])==r['sha256']
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],
                         capture_output=True,text=True)
    assert tests.returncode==0 and 'Ran 14 tests' in tests.stderr
    final_report=root/'心路历程/v3四项评估最终结论_20260908.md'
    evidence[str(final_report.relative_to(root))]=sa.digest(final_report)
    report=dict(status='PASS_EVALUATION_REQUIREMENTS_COMPLETED',requirements={
        '1':'Frozen 365-cell/21-fold error and observable quality diagnosis completed; raw-cache observability limits disclosed.',
        '2':'Ordinary PLS and simple fusion alternatives measured; novelty risk assessed, not declared resolved positively.',
        '3':'Two prospectively bounded probes completed and rejected; original v3 unchanged.',
        '4':'Frozen previously unscored cell-cohort confirmation completed; historical source exposure, small sample and post-score data correction disclosed.'},
        frozen_artifacts_verified=42,unit_tests=14,unit_test_output=tests.stderr,evidence_sha256=evidence,
        remaining_work_within_requested_evaluation=[],
        not_claimed=['Journal acceptance','Established novelty','Universal or practically adequate cross-source accuracy',
                     'A never-before-used laboratory','A wholly blind one-pass dataset-cleaning process'],
        code_sha256=sa.digest(Path(__file__)))
    sa.write_json(HERE/'completion_audit.json',report)
    print(report['status'],flush=True)


if __name__=='__main__':main()
