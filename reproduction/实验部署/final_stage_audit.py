#!/usr/bin/env python3
"""Requirement-by-requirement audit before new-model development."""
import csv,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'开发基线选择依据/results';checks={};fail=['Scientific readiness is not established by this legacy file-count audit. Corrected protocol, faithful baseline implementations, convergence evidence and development/test separation require independent validation.']
def csvcheck(path,label,minrows):
    p=R/path
    if not p.exists():fail.append(label+' missing');return
    with p.open() as f:d=list(csv.DictReader(f))
    finite=True
    for row in d:
        for k,v in row.items():
            if k in {'mae','rmse','macro_mae','macro_rmse'}:
                try:finite &= math.isfinite(float(v))
                except:finite=False
    checks[label]={'rows':len(d),'finite_metrics':finite,'status':'PASS' if len(d)>=minrows and finite else 'FAIL'}
    if len(d)<minrows or not finite:fail.append(label+' incomplete/nonfinite')
csvcheck(Path('formal_protocol_all_10seed.csv'),'raw_10seed_macro',360)
csvcheck(Path('formal_protocol_all_10seed_cells/fair_cell_results.csv'),'raw_10seed_cell',27000)
csvcheck(Path('remaining_baselines_10seed.csv'),'hi_10seed',128)
csvcheck(Path('uda_10seed.csv'),'uda_10seed',30)
csvcheck(Path('maml_10seed.csv'),'maml_10seed',10)
csvcheck(Path('pinn_10seed.csv'),'pinn_10seed',10)
csvcheck(Path('last_block_10seed.csv'),'last_block_10seed',80)
csvcheck(Path('full_target_heldout_upper/upper_results.csv'),'heldout_upper',10)
csvcheck(Path('multidataset_hi_matrix/matrix_results.csv'),'multi_core_external_matrix',150)
csvcheck(Path('self_dataset_hi/self_results.csv'),'within_dataset_controls',45)
csvcheck(Path('matr_cross_hi/matr_results.csv'),'matr_cross_track',10)
csvcheck(Path('legacy_rpt_external/legacy_results.csv'),'legacy_rpt_external',90)
csvcheck(Path('core_p4_p5/p4_p5_results.csv'),'P4_P5_multisource_lodo',30)
csvcheck(Path('xjtu_protocol_holdout/protocol_results.csv'),'P1_protocol_holdout',30)
checks['protocol']=json.loads((ROOT/'实验部署/protocol_validation.json').read_text())['status']=='PROTOCOL_VALID'
checks['deployment_verifier']='NOT_RECHECKED_BY_THIS_SCRIPT'
checks['dependency_lock']=(ROOT/'实验部署/requirements.lock.txt').exists()
status='FINAL_PRE_MODEL_STAGE_VERIFIED' if not fail and all(v is True or (isinstance(v,dict) and v.get('status')=='PASS') or v=='DEPLOYMENT_VERIFIED' for v in checks.values()) else 'INCOMPLETE'
report={'status':status,'checks':checks,'failures':fail,'scope':'baseline/data/protocol stage; proposed model intentionally not implemented'}
(ROOT/'实验部署/final_stage_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps(report,indent=2,ensure_ascii=False));sys.exit(0 if status.startswith('FINAL') else 1)
