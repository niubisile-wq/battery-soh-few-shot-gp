#!/usr/bin/env python3
"""Machine-readable audit of the completed baseline/data stage."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
R=ROOT/'开发基线选择依据/results'; out={"task":"XJTU_to_HUST_K10","checks":{}}
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def check(name,path,method_variants,seeds=5):
    p=R/path
    if not p.exists(): out['checks'][name]={"status":"MISSING","path":str(p)};return
    d=rows(p); got={}
    for r in d: got.setdefault((r.get('model'),r.get('variant')),set()).add(r.get('seed'))
    missing=[f'{m}/{v}' for m,v in method_variants if len(got.get((m,v),set()))<seeds]
    out['checks'][name]={"status":"PASS" if not missing else "INCOMPLETE","rows":len(d),"missing":missing,"path":str(p)}
check('handcrafted_HI','remaining_baselines/remaining_results.csv',[(m,v) for m in ['Ridge','SVR_RBF','RandomForest','XGBoost','PLSR','GPR'] for v in ['source_only','target_only']])
check('raw_last_block_upper','last_block_upper/last_block_upper_results.csv',[(m,'last_block') for m in ['MLP','CNN1D','LSTM','GRU','CNN_LSTM','TCN','Transformer','PatchTSTLite']])
check('maml','maml/maml_results.csv',[('MAML','meta_train_then_K10_adapt')])
check('uda','uda_separate/uda_results.csv',[(m,'unlabeled_target_adaptation') for m in ['MMD','DeepCORAL','DANN']])
check('pinn','pinn4soh_common_input/pinn_results.csv',[('PINN4SOH_common_input','support_adaptation')])
check('raw_main','formal_protocol_all/fair_results.csv',[(m,v) for m in ['MLP','CNN1D','LSTM','GRU','CNN_LSTM','TCN','Transformer','PatchTSTLite'] for v in ['source_only','target_only','head_only','full_finetune']])
out['data_audit']=json.loads((ROOT/'实验部署/data_audit/audit_summary.json').read_text())
out['protocol_validation']=json.loads((ROOT/'实验部署/protocol_validation.json').read_text())
out['governance_warning']={'deployment_manifest_neural_network_seeds':10,'completed_repeated_seeds':5,'meaning':'现有结果可用于内部筛查，但尚未达到部署清单规定的正式10种子门槛'}
out['remaining_known_gaps']=['multi-dataset P1-P6 execution outputs','full-target held-out upper bound (current diagnostic is in-sample oracle)','formal raw main per-cell export','dependency lock and final deployment verifier']
(ROOT/'实验部署/current_step_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(out,indent=2,ensure_ascii=False))
