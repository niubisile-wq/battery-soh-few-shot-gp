#!/usr/bin/env python3
"""Merge seed 0-4 and 5-9 batches without silently dropping rows."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'开发基线选择依据/results'
def merge(rel, a, b, out):
    rows=[]
    for p in (R/a,R/b):
        with p.open() as f: rows.extend(csv.DictReader(f))
    with (R/out).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    return len(rows)
out={}
out['raw']=merge('', 'formal_protocol_all/fair_results.csv','formal_protocol_all_seed5_9/fair_results.csv','formal_protocol_all_10seed.csv')
out['hi']=merge('', 'remaining_baselines/remaining_results.csv','remaining_baselines_seed5_9/remaining_results.csv','remaining_baselines_10seed.csv')
out['uda']=merge('', 'uda_separate/uda_results.csv','uda_separate_seed5_9/uda_results.csv','uda_10seed.csv')
out['maml']=merge('', 'maml/maml_results.csv','maml_seed5_9/maml_results.csv','maml_10seed.csv')
out['pinn']=merge('', 'pinn4soh_common_input/pinn_results.csv','pinn4soh_common_input_seed5_9/pinn_results.csv','pinn_10seed.csv')
out['last_block']=merge('', 'last_block_upper/last_block_upper_results.csv','last_block_seed5_9/last_block_upper_results.csv','last_block_10seed.csv')
(ROOT/'实验部署/seed_merge_manifest.json').write_text(json.dumps({'seed_batches':['0-4','5-9'],'outputs':out},indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
