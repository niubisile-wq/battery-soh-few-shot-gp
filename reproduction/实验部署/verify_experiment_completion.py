#!/usr/bin/env python3
"""Strict completion gate for the pre-new-model baseline/data stage."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'开发基线选择依据/results'; failures=[]
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def seeds(path, keys):
    d=read(path); got={}
    for r in d:got.setdefault(tuple(r[k] for k in keys),set()).add(int(r['seed']))
    return d,got
def require(path, label, expected=None):
    if not path.exists(): failures.append(f'{label}: missing {path}');return None
    d=read(path)
    if not d: failures.append(f'{label}: empty');return None
    if expected:
        _,got=seeds(path,expected[0])
        for key in expected[1]:
            if len(got.get(tuple(key),set()))<10:failures.append(f'{label}: {key} has {sorted(got.get(tuple(key),set()))}, need seeds 0-9')
    return d
raw=require(R/'formal_protocol_all_10seed.csv','raw_10seed',(['model','variant'],[*( (m,v) for m in ['MLP','CNN1D','LSTM','GRU','CNN_LSTM','TCN','Transformer','PatchTSTLite'] for v in ['source_only','target_only','head_only','full_finetune']),('SVR_RBF','source_only'),('SVR_RBF','target_only'),('XGBoost','source_only'),('XGBoost','target_only')]))
for fn,label in [('remaining_baselines_10seed.csv','HI_10seed'),('uda_10seed.csv','UDA_10seed'),('maml_10seed.csv','MAML_10seed'),('pinn_10seed.csv','PINN_10seed'),('last_block_10seed.csv','last_block_10seed')]:
    require(R/fn,label)
if not (R/'full_target_heldout_upper/upper_results.csv').exists():failures.append('heldout upper bound missing')
mp=R/'multidataset_hi_matrix/matrix_results.csv'
if not mp.exists() or len(read(mp))<150:failures.append('multi-dataset matrix incomplete')
sp=R/'self_dataset_hi/self_results.csv'
if not sp.exists() or len(read(sp))<45:failures.append('within-dataset control incomplete')
matp=R/'matr_cross_hi/matr_results.csv'
if not matp.exists() or len(read(matp))<10:failures.append('MATR cross-dataset track incomplete')
if not (R/'formal_protocol_all_10seed_cells/fair_cell_results.csv').exists():failures.append('raw formal per-cell results missing')
if not (ROOT/'实验部署/requirements.lock.txt').exists():failures.append('dependency lock missing')
if failures:
    print('EXPERIMENT_INCOMPLETE');print('\n'.join(failures));sys.exit(1)
print('EXPERIMENT_STAGE_VERIFIED')
