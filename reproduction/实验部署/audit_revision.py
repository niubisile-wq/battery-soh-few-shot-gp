"""Read-only checks of input evidence and manuscript consistency.

Only this revision's audit report and PDF page previews are written.
No model imports, fitting, re-tuning or prediction generation is performed.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import re
import numpy as np
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent
DOC = ROOT/'正文'
EVID = DOC/'evidence'
CHECK = ROOT/'核验'
CHECK.mkdir(exist_ok=True)
DATASETS = ['XJTU','MATR','Tongji']
report = {'scope':'manuscript revision only','model_training_performed':False,'checks':{}}


def digest(path):
    h=hashlib.sha256()
    path=Path(path)
    if not path.exists():
        moved = {
            '/root/autodl-tmp/main.tex': ROOT.parent/'参考材料/main.tex',
            '/root/autodl-tmp/M4.pdf': ROOT.parent/'参考材料/M4.pdf',
            '/root/autodl-tmp/1-s2.0-S0952197626011176-main.pdf': ROOT.parent/'参考材料/1-s2.0-S0952197626011176-main.pdf',
            '/root/autodl-tmp/补充材料.zip': ROOT.parent/'参考材料/补充材料.zip',
            '/root/autodl-tmp/论文修订版_实验问题修复': ROOT.parent/'归档_相关材料/论文修订版_实验问题修复',
            '/root/autodl-tmp/supplement_review_20260910': ROOT.parent/'归档_相关材料/supplement_review_20260910',
            '/root/autodl-tmp/临时_已停止的神经补跑': ROOT.parent/'归档_相关材料/临时_已停止的神经补跑',
        }
        for old, new in moved.items():
            if str(path) == old or str(path).startswith(old + '/'):
                path = Path(str(new) + str(path)[len(old):])
                break
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    with (EVID/path).open() as f:return list(csv.DictReader(f))


def check(key, value, detail):
    report['checks'][key]={'passed':bool(value),'detail':detail}
    if not value:
        (CHECK/'revision_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        raise AssertionError((key,detail))


# Narrative-only revision: retain the previous version and every numerical asset.
previous = ROOT.parent/'论文修订版_证据收束_20260911'
snapshot = json.loads((ROOT/'previous_version_manifest.json').read_text())
bad = [p for p, h in snapshot.items() if not Path(p).is_file() or digest(p) != h]
check('previous_version_unchanged', not bad, {'files':len(snapshot), 'mismatches':bad})
assets = []
for folder in ['tables', 'evidence', 'figures_base', 'figures_supplement', 'figures_new']:
    old_paths = {p.relative_to(previous/'正文') for p in (previous/'正文'/folder).rglob('*') if p.is_file()}
    new_paths = {p.relative_to(DOC) for p in (DOC/folder).rglob('*') if p.is_file()}
    check('asset_inventory_'+folder, old_paths == new_paths, {'previous':len(old_paths), 'current':len(new_paths)})
    assets.extend(old_paths)
assets.append(Path('bibliography.tex'))
label_only = Path('tables/controls.tex')
bad = [str(p) for p in assets if p != label_only and digest(previous/'正文'/p) != digest(DOC/p)]
check('numerical_assets_and_references_unchanged', not bad, {'files':len(assets), 'mismatches':bad, 'label_only_table':str(label_only)})
def table_numbers(path):
    s=Path(path).read_text()
    body=s.split('\\begin{tabular}',1)[1].split('\\end{tabular}',1)[0]
    return re.findall(r'(?<![A-Za-z])\d+\.\d+', body)
check('controls_table_values_unchanged', table_numbers(previous/'正文'/label_only) == table_numbers(DOC/label_only), {'entries':len(table_numbers(DOC/label_only))})
old_main = (previous/'正文/main.tex').read_text()
new_main = (DOC/'main.tex').read_text()
# Compare complete blocks, not just their environment names.
math_blocks = lambda s: [m.group(0) for m in re.finditer(r'\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}', s, re.S)]
check('display_equations_unchanged', math_blocks(old_main) == math_blocks(new_main), {'blocks':len(math_blocks(new_main))})

# Copy integrity and the pre-existing protected manuscript/model/source files.
manifest=json.loads((ROOT/'evidence_manifest.json').read_text())['copied_files']
bad=[r['copy'] for r in manifest if digest(ROOT/r['copy'])!=r['sha256'] or digest(r['source'])!=r['sha256']]
check('evidence_copies',not bad,{'files':len(manifest),'mismatches':bad})
preservation=json.loads((ROOT.parent/'补充实验_20260910/preservation.json').read_text())['protected_files']
bad=[p for p,h in preservation.items() if digest(p)!=h]
check('pre_existing_protected_files_unchanged',not bad,{'files':len(preservation),'mismatches':bad})
core=ROOT.parent/'模块研发/results/m4_random_function_candidate_v1'
frozen=json.loads((core/'candidate.json').read_text())
bad=[r['artifact'] for r in frozen['manifest'] if digest(core/r['artifact'])!=r['sha256']]
check('frozen_model_adapters_unchanged',not bad,{'adapters':len(frozen['manifest']),'mismatches':bad})
bad=[r['path'] for r in frozen['data'] if digest(ROOT.parent/r['path'])!=r['sha256']]
check('frozen_input_data_unchanged',not bad,{'files':len(frozen['data']),'mismatches':bad})
statmanifest=json.loads((EVID/'statistics/manifest.json').read_text())
for key in ['prediction_sha256']:
    bad=[p for p,h in statmanifest[key].items() if digest(p)!=h]
    check(key+'_unchanged',not bad,{'files':len(statmanifest[key]),'mismatches':bad})

# Independently reaggregate cell metrics (not model predictions).
cells=read('statistics/cell_metrics.csv')
ab={(r['dataset'],r['group']):r for r in read('statistics/full_ablation_16_groups.csv')}
ci={(r['dataset'],r['cell_id'],r['group']):r for r in cells}
maxerr=0.
for (ds,g),r in ab.items():
    selected=[x for x in cells if x['dataset']==ds and x['group']==g]
    assert len(selected)==int(r['cells'])
    for m in ['mae','rmse','p95_ae','low_mae','low_rmse','low_p95_ae']:
        bydomain=defaultdict(list)
        for c in selected:
            if c[m] and np.isfinite(float(c[m])):bydomain[c['domain']].append(float(c[m]))
        val=np.mean([np.mean(v) for v in bydomain.values()])
        maxerr=max(maxerr,abs(val-float(r[m])))
check('cell_domain_reaggregation',maxerr<1e-10,{'cell_rows':len(cells),'aggregate_rows':len(ab),'max_abs_pp':maxerr})
edges=[]
for ds,g in ab:
    for module in '1234':
        if module not in g:
            child='B'+''.join(sorted(g[1:]+module))
            edges.append(all(float(ab[(ds,child)][m])<float(ab[(ds,g)][m]) for m in ['mae','rmse','p95_ae']))
check('32_directional_checks',len(edges)==96 and all(edges),{'module_edges':32,'dataset_edge_checks':len(edges)})

# Bind the old plotted summary numbers to the newly reaggregated evidence.
maxerr=0.
for r in json.loads((EVID/'figure_data/01_main_metrics_data.json').read_text()):
    for ds,val in zip(DATASETS,r['values']):maxerr=max(maxerr,abs(val-float(ab[(ds,r['group'])][r['metric']])))
heat=json.loads((EVID/'figure_data/02_ablation_heatmap_data.json').read_text())
for j,g in enumerate(heat['groups']):
    for k,ds in enumerate(heat['datasets']):
        value=float(ab[(ds,g)]['mae']);base=float(ab[(ds,'B')]['mae'])
        maxerr=max(maxerr,abs(value-heat['mae'][j][k]),abs(100*(base-value)/base-heat['reduction_percent'][j][k]))
for name in ['03_paired_cell_errors','07_m4_increment_detail']:
    for r in json.loads((EVID/'figure_data'/f'{name}_data.json').read_text()):
        parent=r.get('parent','B123')
        for cid,val in zip(r['cell_ids'],r['delta_pp']):
            cid=Path(cid).name
            delta=float(ci[(r['dataset'],cid,'B1234')]['mae'])-float(ci[(r['dataset'],cid,parent)]['mae'])
            maxerr=max(maxerr,abs(delta-val))
check('reused_metric_and_paired_plot_data',maxerr<1e-8,{'max_abs_pp_or_percent':maxerr})

combined=read('baselines/summary/combined_12_baselines_plus_final.csv')
neural=read('baselines/summary/comparison.csv')
check('baseline_coverage',len(combined)==39 and len(neural)==27,{'combined_rows':len(combined),'neural_settings_rows':len(neural)})
for ds in DATASETS:
    full=float(ab[(ds,'B1234')]['mae'])
    check('reported_primary_MAE_'+ds,all(full<float(r['mae_mean_pp']) for r in combined if r['dataset']==ds and r['method']!='B1234_frozen'),{'frozen_MAE_pp':full})
lower=next(r for r in neural if r['dataset']=='Tongji' and r['model']=='MAML_MLP' and r['setting']=='source_bias')
check('retained_losing_P95_comparison',float(lower['p95_ae_mean_pp'])<float(ab[('Tongji','B1234')]['p95_ae']),{'MAML_bias_P95':lower['p95_ae_mean_pp'],'frozen_P95':ab[('Tongji','B1234')]['p95_ae']})

ecells=read('external/confirmation/summary/cell_mean_across_sources.csv')
esummary={r['group']:r for r in read('external/confirmation/summary/cohort_summary.csv')}
maxerr=0.
for g,r in esummary.items():
    for m in ['mae_pp','rmse_pp','p95_ae_pp']:
        bydomain=defaultdict(list)
        for c in ecells:
            if c['group']==g:bydomain[c['domain']].append(float(c[m]))
        maxerr=max(maxerr,abs(np.mean([np.mean(v) for v in bydomain.values()])-float(r[m])))
check('confirmation_group_aggregation',maxerr<1e-10,{'max_abs_pp':maxerr,'cells':7,'protocol_groups':4})
check('negative_confirmation_retained',float(esummary['B1234']['mae_pp'])>float(esummary['B']['mae_pp']),{'B_MAE':esummary['B']['mae_pp'],'B1234_MAE':esummary['B1234']['mae_pp']})

# Typesetting and cross-reference checks against final compilation.
log=(DOC/'main.log').read_text(errors='replace')
badlines=[line for line in log.splitlines() if any(x in line for x in ['undefined','multiply defined','Overfull','Float too large','! LaTeX Error','Fatal error','destination with the same identifier'])]
check('latex_no_unresolved_or_overflow_errors',not badlines,{'matching_lines':badlines})
texfiles=list(DOC.glob('*.tex'))+list((DOC/'tables').glob('*.tex'))
text='\n'.join(p.read_text() for p in texfiles)
labels=re.findall(r'\\label\{([^}]+)\}',text)
keys=re.findall(r'\\bibitem\{([^}]+)\}',text)
check('unique_labels_and_bibliography',len(labels)==len(set(labels)) and len(keys)==len(set(keys)),{'labels':len(labels),'reference_entries':len(keys)})
pdf=fitz.open(DOC/'main.pdf')
outside=[]
pages=[]
for i,page in enumerate(pdf):
    txt=page.get_text()
    snippets=re.findall(r'(?:Table (?:S)?\d+|Figure (?:S)?\d+):[^\n]+',txt)
    pages.append({'pdf_page':i+1,'caption_starts':snippets,'words':len(page.get_text('words'))})
    for w in page.get_text('words'):
        if w[0]<0 or w[1]<0 or w[2]>page.rect.width+1 or w[3]>page.rect.height+1:outside.append((i+1,w[4]))
check('pdf_text_inside_page',not outside,{'outside':outside,'pages':len(pdf)})
report['pdf_pages']=pages
report['artifacts']={'main_pdf_sha256':digest(DOC/'main.pdf'),'main_tex_sha256':digest(DOC/'main.tex'),'generated_tables':len(list((DOC/'tables').glob('*.tex')))}
previews=CHECK/'页面预览'
previews.mkdir(exist_ok=True)
for i in [0,3,6,12,14,15,19,28,29,33,43,48]:
    if i<len(pdf):pdf[i].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(previews/f'page_{i+1:02d}.png')
report['status']='PASS'
(CHECK/'revision_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(report['checks']),'pdf_pages':len(pdf),'generated_tables':report['artifacts']['generated_tables'],'model_training_performed':False}))
