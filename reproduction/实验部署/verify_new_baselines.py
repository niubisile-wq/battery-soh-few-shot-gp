from pathlib import Path
import csv,json,numpy as np
ROOT=Path(__file__).resolve().parent
checks=[]
for name,expected in [('at_gpr_v1',365),('transfer_stacking_svr_v1',365),('enhanced_maml_anil_v1',3650)]:
 p=ROOT/name; man=json.loads((p/'manifest.json').read_text()); rows=list(csv.DictReader((p/'per_cell_results.csv').open()))
 assert len(rows)==expected,(name,len(rows)); assert len(list((p/'predictions').glob('*.npz')))==expected
 assert man.get('K')==10; assert man.get('target_query_labels_used_for_fitting',False) is False; assert man.get('target_query_labels_used_for_adaptation',False) is False
 if name=='enhanced_maml_anil_v1': assert sorted(set(r['seed'] for r in rows))==['0','1','2','3','4']; assert sorted(set(r['method'] for r in rows))==['ANIL','MAML']
 for r in rows:
  z=np.load(p/r['prediction_file'].replace(str(p)+'/','')) if not Path(r['prediction_file']).is_absolute() else np.load(r['prediction_file'])
  assert z['pred'].shape==z['y'].shape and z['pred'].ndim==1 and np.isfinite(z['pred']).all() and np.isfinite(z['y']).all()
 checks.append({'method':name,'rows':len(rows),'prediction_files':len(list((p/'predictions').glob('*.npz'))),'status':'PASS'})
# The source-only GPR companion shares the AT-GPR feature and source protocol
# but keeps a separate archive and manifest.
p=ROOT/'at_gpr_v1'
rows=list(csv.DictReader((p/'gpr_source_per_cell.csv').open()))
man=json.loads((p/'gpr_source_manifest.json').read_text())
assert len(rows)==365 and man['K']==10 and man['target_query_labels_used_for_fitting'] is False
assert len(list((p/'gpr_source_predictions').glob('*.npz')))==365
checks.append({'method':'GPR','rows':len(rows),'prediction_files':len(list((p/'gpr_source_predictions').glob('*.npz'))),'status':'PASS'})
(ROOT/'verification_new_baselines.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2),encoding='utf8');print(json.dumps(checks,indent=2))
