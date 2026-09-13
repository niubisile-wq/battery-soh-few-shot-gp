from pathlib import Path
import csv,json,statistics
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'模块研发/results/m4_global_sensitivity_v1/summary.json'
CAND=json.loads(SRC.read_text())['table']
DATASETS=['XJTU','MATR','Tongji']
# Candidate architecture families and global multipliers already have frozen predictions.
rows=[r for r in CAND if r['group'].startswith('B1234__')]
# Keep only complete candidates (all three datasets) and parse family/step.
candidates={}
for r in rows:
    _,family,step=r['group'].split('__')
    candidates.setdefault((family,float(step)),{})[r['dataset']]=r
candidates={k:v for k,v in candidates.items() if set(v)==set(DATASETS)}
# Existing frozen base and strict few-shot baselines.
frozen=json.loads((ROOT/'模块研发/results/m4_random_function_candidate_v1/candidate.json').read_text())['table']
base={r['dataset']:r for r in frozen if r['group']=='B'}
strict=list(csv.DictReader((ROOT/'新增公平基线/strict_main_baseline_comparison.csv').open()))
strict_by={(r['method'],r['dataset']):r for r in strict}
results=[];choices=[]
for heldout in DATASETS:
    train=[d for d in DATASETS if d!=heldout]
    scored=[]
    for (family,step),byds in candidates.items():
        mean=sum(float(byds[d]['mae']) for d in train)/len(train)
        scored.append((mean,family,step))
    scored.sort(key=lambda x:(x[0],x[1],x[2]))
    mean,family,step=scored[0];candidate=candidates[(family,step)]
    choices.append({'heldout_dataset':heldout,'selection_datasets':train,'chosen_family':family,'chosen_global_step':step,'selection_mean_mae_pp':mean,'candidate_count':len(scored),'top5':[{'selection_mean_mae_pp':x[0],'family':x[1],'global_step':x[2]} for x in scored[:5]]})
    r=candidate[heldout]
    results.append({'heldout_dataset':heldout,'selection_datasets':'+'.join(train),'method':'LODO_selected_B1234','architecture_family':family,'global_step':step,'mae_pp':r['mae'],'rmse_pp':r['rmse'],'p95_ae_pp':r['p95_ae'],'selection_mean_mae_pp':mean})
    b=base[heldout]
    results.append({'heldout_dataset':heldout,'selection_datasets':'+'.join(train),'method':'B','architecture_family':'base','global_step':'','mae_pp':b['mae'],'rmse_pp':b['rmse'],'p95_ae_pp':b['p95_ae'],'selection_mean_mae_pp':''})
    for method in ['GPR','AT-GPR','MAML','ANIL','transfer-stacking-SVR','MAML_MLP','PatchTSTLite','TCN']:
        q=strict_by.get((method,heldout))
        if q:
            results.append({'heldout_dataset':heldout,'selection_datasets':'+'.join(train),'method=method':method,'method':method,'architecture_family':'strict_baseline','global_step':'','mae_pp':q['mae_mean_pp'],'rmse_pp':q['rmse_mean_pp'],'p95_ae_pp':q['p95_ae_mean_pp'],'selection_mean_mae_pp':''})
# Clean accidental field and write.
for r in results:r.pop('method=method',None)
with (ROOT/'留一数据集选择验证_20260913/lodo_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['heldout_dataset','selection_datasets','method','architecture_family','global_step','mae_pp','rmse_pp','p95_ae_pp','selection_mean_mae_pp']);w.writeheader();w.writerows(results)
(ROOT/'留一数据集选择验证_20260913/lodo_selection.json').write_text(json.dumps({'status':'COMPLETE','candidate_source':str(SRC.relative_to(ROOT)),'candidate_count':len(candidates),'selection_metric':'mean MAE over the two non-held-out datasets','choices':choices,'heldout_evaluation_uses_no_selection_labels':True,'limitations':['candidate predictions were frozen before this audit; candidate-generation screens historically used development datasets','this is a selection-level LODO audit, not a fresh nested retraining of every architecture and source fit']},indent=2),encoding='utf8')
print(json.dumps(choices,indent=2))
