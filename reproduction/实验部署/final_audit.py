"""Read-only old-artifact verification; write study-local final acceptance."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
from study import HERE,ROOT,BUNDLE,digest,write_json,preservation


def main():
    protected=preservation();manifest=json.loads((BUNDLE/'candidate.json').read_text());checked={}
    for r in manifest['manifest']:
        path=BUNDLE/r['artifact'];assert digest(path)==r['sha256'];checked[str(path)]=r['sha256']
    data={r['path']:r['sha256'] for r in manifest['data']}
    for p,h in data.items():assert digest(ROOT/p)==h
    for r in manifest['predictions']:assert digest(BUNDLE/r['path'])==r['sha256']
    stats=json.loads((HERE/'statistics/manifest.json').read_text())
    assert stats['status']=='RECOMPUTED_FROZEN_STATISTICS' and stats['cell_rows']==5840 and stats['paired_rows']==144
    assert digest(HERE/'derive_statistics.py')==stats['code_sha256']
    for name,h in stats['outputs'].items():assert digest(HERE/'statistics'/name)==h
    baseline=json.loads((HERE/'baselines/summary/checkpoint_replay_audit.json').read_text())
    assert baseline['status']=='ALL_CHECKPOINTS_AND_SOURCE_VALIDATION_VERIFIED' and baseline['seed_tasks']==189
    assert digest(HERE/'verify_baselines.py')==baseline['code_sha256']
    for p,h in baseline['input_hashes'].items():assert digest(p)==h
    bsummary=json.loads((HERE/'baselines/summary/audit.json').read_text())
    assert bsummary['complete_seed_tasks']==189 and not bsummary['missing']
    for p,h in bsummary['complete_hashes'].items():assert digest(p)==h
    external=json.loads((HERE/'external/confirmation/summary/audit.json').read_text())
    assert external['status']=='FROZEN_CONFIRMATION_VERIFIED' and external['cell_group_source_rows']==210
    assert digest(HERE/'summarize_confirmation.py')==external['code_sha256']
    for p,h in external['input_artifact_sha256'].items():assert digest(p)==h
    for name,h in external['output_sha256'].items():assert digest(HERE/'external/confirmation/summary'/name)==h
    gate=json.loads((HERE/'external/confirmation_data_gate.json').read_text())
    for p,h in gate['input_hashes'].items():assert digest(p)==h
    assert digest(HERE/'verify_confirmation_data.py')==gate['verifier_sha256']
    request=json.loads((HERE/'external/confirmation/request.json').read_text())
    assert digest(HERE/'external/confirmation_data_gate.json')==request['gate_sha256']
    assert digest(HERE/'run_confirmation.py')==request['code_sha256']
    legacy=json.loads((HERE/'baselines/summary/legacy_aggregation_audit.json').read_text())
    assert legacy['max_error_ratio']<1e-12 and legacy['summary_rows']==27
    assert digest(HERE/'consolidate_baselines.py')==legacy['code_sha256']
    for p,h in legacy['source_sha256'].items():assert digest(p)==h
    for name,h in legacy['output_sha256'].items():assert digest(HERE/'baselines/summary'/name)==h
    report_file=HERE/'补充实验结果与结论.md'
    assert report_file.exists() and report_file.stat().st_size>1000
    import re
    for target in re.findall(r'\]\(([^)]+)\)',report_file.read_text()):
        if not target.startswith('https://'):assert (report_file.parent/target).exists(),target
    # No pre-existing manuscript/model outputs are created by the test runner.
    previous=HERE/'test_results.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old['returncode']!=0:write_json(HERE/'test_history'/f'{time.time_ns()}.json',old)
    tests=subprocess.run([sys.executable,'-B','-m','unittest','test_statistics','test_measurement','test_confirmation_data','-v'],
        cwd=HERE,text=True,capture_output=True)
    write_json(HERE/'test_results.json',dict(returncode=tests.returncode,stdout=tests.stdout,stderr=tests.stderr))
    print(tests.stderr,flush=True);assert tests.returncode==0
    preservation()
    report=dict(status='SUPPLEMENTAL_EXPERIMENTS_COMPLETE_AND_VERIFIED',paper_edited=False,core_model_edited=False,
        protected_original_files=len(protected['protected_files']),frozen_adapters_verified=len(checked),
        original_prediction_files_verified=len(manifest['predictions']),original_data_files_verified=len(data),
        statistics_cell_rows=5840,paired_statistics_rows=144,baseline_seed_tasks=189,
        baseline_cell_seed_predictions_replayed=3285,confirmation_unique_cells=7,confirmation_measured_pairs=5633,
        confirmation_unique_queries=5563,confirmation_cell_group_source_predictions=210,
        acceptance='Completion means outputs and information boundaries verified; does not require positive model results',
        finding='Development gains remain. Frozen seven-cell CALCE confirmation does not show overall B1234 improvement over B; retain negative results and do not retune on this cohort.',
        original_adapter_hashes=checked,script_sha256={p.name:digest(p) for p in HERE.glob('*.py')},
        report_sha256=digest(report_file),report=str(report_file),
        evidence_manifests={str(p):digest(p) for p in [HERE/'statistics/manifest.json',HERE/'baselines/summary/audit.json',
            HERE/'baselines/summary/checkpoint_replay_audit.json',HERE/'external/confirmation/summary/audit.json',
            HERE/'external/confirmation_data_gate.json',HERE/'baselines/summary/legacy_aggregation_audit.json',HERE/'test_results.json']})
    write_json(HERE/'completion_audit.json',report)
    print('SUPPLEMENTAL_EXPERIMENTS_COMPLETE_AND_VERIFIED',len(checked),'frozen adapters unchanged',flush=True)


if __name__=='__main__':main()
