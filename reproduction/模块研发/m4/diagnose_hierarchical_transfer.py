"""Retrospective validation/query transfer, no configuration selection."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa,HERE
from evaluate import dump_csv


def main():
    screen=HERE.parent/'results/m4_hierarchical_screen_v1'
    audit=json.loads((screen/'verification.json').read_text())
    assert audit['status']=='PASS' and audit['summary_sha256']==sa.digest(screen/'summary.json')
    frozen=json.loads((screen/'frozen_selections.json').read_text())
    with (screen/'cells.csv').open() as f:query=list(csv.DictReader(f))
    def macro(rows):
        buckets=defaultdict(list)
        for r in rows:buckets[r['domain']].append(float(r['mae']))
        assert buckets
        return float(np.mean([np.mean(v) for v in buckets.values()]))
    rows=[]
    for s in frozen['selections']:
        root=Path(s['branch_root']);assert sa.digest(root/'result.json')==s['result_sha256']
        result=json.loads((root/'result.json').read_text());ds,domain=s['fold'].split(':')
        for family in ('hierarchical','cell_only'):
            chosen=s['choices'][family]['selected']
            for parent in ('B','B1','B2','B3','B12','B13','B23','B123'):
                vr=[r for r in result['rows'] if r['group']==parent and r['key']==chosen['key']]
                vb=100*macro([r for r in vr if r['gain']==0])
                va=100*macro([r for r in vr if r['gain']==chosen['gain']])
                child=parent+'4'+('_cell_only' if family=='cell_only' else '')
                qr=[r for r in query if r['dataset']==ds and r['domain']==domain]
                qb=macro([r for r in qr if r['group']==parent]);qa=macro([r for r in qr if r['group']==child])
                rows.append(dict(fold=s['fold'],family=family,parent=parent,key=chosen['key'],gain=chosen['gain'],
                    validation_before_pp=vb,validation_delta_pp=va-vb,query_before_pp=qb,query_delta_pp=qa-qb))
    assert len(rows)==336
    focus=[r for r in rows if r['family']=='hierarchical' and r['parent']=='B123']
    reversals=[r for r in focus if r['validation_delta_pp']< -1e-10 and r['query_delta_pp']>1e-10]
    out=HERE.parent/'results/m4_hierarchical_transfer_diagnosis_v1';out.mkdir(exist_ok=False)
    dump_csv(out/'folds.csv',rows)
    sa.write_json(out/'summary.json',dict(status='RETROSPECTIVE_DIAGNOSIS_ONLY',rows=336,
        full_parent=focus,validation_improves_query_worsens=reversals,
        inactive_folds=[r['fold'] for r in focus if r['gain']==0],
        source_sha256=sa.digest(screen/'summary.json'),audit_sha256=sa.digest(screen/'verification.json'),
        code_sha256=sa.digest(Path(__file__)),
        limits='Original validation and repeatedly exposed query cohorts. Diagnoses sign mismatch,not cause. No oracle selection,no new candidate.'))
    print('inactive',[r['fold'] for r in focus if r['gain']==0],flush=True)
    print('validation/query reversals',reversals,flush=True)


if __name__=='__main__':main()
