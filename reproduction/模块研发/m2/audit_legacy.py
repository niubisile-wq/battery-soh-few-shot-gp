"""Produce concrete evidence for withdrawing the legacy M2 acceptance."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'m1'))
from data import ROOT,write_json


def main():
    val=defaultdict(set); test=defaultdict(set); files={}; physical=[]
    for path in sorted((ROOT/'模块研发/results/m2_prediction_cache_v1').glob('*.json')):
        rows=json.loads(path.read_text()); files[path.stem]=rows
        for r in rows:
            (val if r['split']=='val' else test)[r['dataset']].add(r['cell_id'])
        old_job=ROOT/'模块研发/results/m1_v1/screen_v1/jobs'/f'P02_anchor_physics__{path.stem}'
        chosen=json.loads((old_job/'tuning.json').read_text())['chosen']
        saved=json.loads((old_job/'cell_results.json').read_text())
        bycell=defaultdict(list)
        for r in rows:
            if r['split']=='test': bycell[r['cell_id']].append(r)
        errors=[]
        for r in saved:
            rr=sorted(bycell[r['cell_id']],key=lambda z:z['q'])
            with np.load(ROOT/r['prediction_file']) as z:
                assert np.array_equal(z['y'],np.asarray([x['y'] for x in rr]))
                diff=np.asarray([x['pb'] for x in rr])-z['pred']
            errors.append(float(np.max(abs(diff))))
        physical.append({'fold':path.stem,'mode':chosen['mode'],
                         'max_cached_physical_prediction_error':max(errors),
                         'cells_affected':sum(e>1e-8 for e in errors)})
    overlap={d:{'cells':sorted(val[d]&test[d]),'count':len(val[d]&test[d])} for d in sorted(val)}
    cal=json.loads((ROOT/'模块研发/results/m2_calibration_support_v3/summary.json').read_text())
    budget=[]
    for f in cal['folds']:
        job=ROOT/'模块研发/results/m1_v1/screen_v2/jobs'/('P07_pls_concat__'+f['fold'].replace(':','__'))
        n=len(json.loads((job/'provenance.json').read_text())['source_keys'])
        budget.append({'fold':f['fold'],'allowed_total_source_labels':n,
                       'legacy_calibration_query_labels':f['meta_source_rows'],
                       'exceeds_total_allowed_budget':f['meta_source_rows']>n})
    result={'status':'LEGACY_ACCEPTANCE_WITHDRAWN','pooled_validation_test_overlap':overlap,
            'source_budget_audit':budget,'physical_replay':physical,
            'affected_physical_folds':sum(x['cells_affected']>0 for x in physical),
            'affected_physical_cells':sum(x['cells_affected'] for x in physical)}
    write_json(ROOT/'模块研发/results/m2_strict_ablation_v1/legacy_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_budget_audit','physical_replay']},ensure_ascii=False))


if __name__=='__main__':main()
