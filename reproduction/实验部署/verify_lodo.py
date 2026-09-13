from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'留一数据集选择验证_20260913';d=json.loads((p/'lodo_selection.json').read_text());rows=list(csv.DictReader((p/'lodo_results.csv').open()))
assert d['status']=='COMPLETE' and d['candidate_count']==35 and len(d['choices'])==3
for c in d['choices']:
 assert c['heldout_dataset'] not in c['selection_datasets']
 assert len(c['selection_datasets'])==2
 assert c['chosen_global_step'] in [0.1,0.25,0.5,0.75,1.0]
 q=next(r for r in rows if r['heldout_dataset']==c['heldout_dataset'] and r['method']=='LODO_selected_B1234')
 b=next(r for r in rows if r['heldout_dataset']==c['heldout_dataset'] and r['method']=='B')
 assert float(q['mae_pp'])<float(b['mae_pp'])
print({'status':'PASS','choices':[(c['heldout_dataset'],c['chosen_family'],c['chosen_global_step']) for c in d['choices']]})
(p/'verification.json').write_text(json.dumps({'status':'PASS','choice_count':3,'heldout_not_in_selection':True,'lodo_b_mae_all_improved':True},indent=2))
