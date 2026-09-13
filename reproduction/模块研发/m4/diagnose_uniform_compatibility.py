"""Compare selected full-parent validation gains across parent combinations.

Query values are retrospective diagnostics, never selection or deployment gates.
"""
import csv,json
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa,HERE
from score import PARENTS
from evaluate import dump_csv


def main():
    results=HERE.parent/'results';screen=results/'m4_evidence_screen_v1';valroot=results/'m4_evidence_validation_v1'
    verification=json.loads((screen/'verification.json').read_text());assert verification['status']=='PASS'
    assert verification['summary_sha256']==sa.digest(screen/'summary.json')
    frozen=json.loads((screen/'frozen_selections.json').read_text())
    valcells={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')}
    # Already-scored raw and0.1 sensitivity, no new candidate search.
    raw=list(csv.DictReader((screen/'cells.csv').open()))
    small=list(csv.DictReader((results/'m4_evidence_uniform_sensitivity_v1/cells.csv').open()))
    lookup={(r['cell_id'],r['group']):r for r in raw};small_lookup={(r['cell_id'],r['group']):r for r in small}
    rows=[]
    for selection in frozen['selections']:
        fold=selection['fold'];ds,domain=fold.split(':');folder=Path(selection['validation_root'])
        r=json.loads((folder/'result.json').read_text());choice=selection['choices']['uniform']['selected']
        assert r['selected']['uniform']==choice
        stored=joblib.load(folder/'predictions.joblib');vp=joblib.load(r['parent_path'])
        _,val,test=sa.split(valcells[ds],fold)
        for step in (1.,.1):
            for group in PARENTS:
                before={};after={}
                for c in val:
                    p=vp[c.id,group];q=stored['predictions'][c.id,choice['key']]
                    candidate=p+step*choice['gain']*(q-p)
                    before.setdefault(c.domain,[]).append(float(np.mean(abs(p-c.y[sa.K:])))*100)
                    after.setdefault(c.domain,[]).append(float(np.mean(abs(candidate-c.y[sa.K:])))*100)
                vbefore=float(np.mean([np.mean(v) for v in before.values()]));vafter=float(np.mean([np.mean(v) for v in after.values()]))
                parentmae=float(np.mean([float(lookup[c.id,group]['mae']) for c in test]))
                if step==1:childmae=float(np.mean([float(lookup[c.id,group+'4_uniform']['mae']) for c in test]))
                else:childmae=float(np.mean([float(small_lookup[c.id,group+'4__0.1']['mae']) for c in test]))
                rows.append(dict(fold=fold,dataset=ds,domain=domain,parent=group,step=step,gain=choice['gain'],key=choice['key'],
                    validation_before_pp=vbefore,validation_delta_pp=vafter-vbefore,query_before_pp=parentmae,query_delta_pp=childmae-parentmae))
    assert len(rows)==336
    out=results/'m4_uniform_compatibility_diagnosis_v1';out.mkdir(exist_ok=False)
    dump_csv(out/'folds.csv',rows)
    tradeoffs=[r for r in rows if r['step']==1 and r['parent']=='B3' and r['validation_delta_pp']>1e-10]
    sa.write_json(out/'summary.json',dict(status='RETROSPECTIVE_COMPATIBILITY_DIAGNOSIS',rows=336,
        B3_validation_regressions=tradeoffs,code_sha256=sa.digest(Path(__file__)),
        source_sha256=sa.digest(screen/'summary.json'),
        limits='No candidate or new gate selected. Query truth is retrospective. Validation mismatch may motivate a selection control,not prove cause or guarantee transfer.'))
    print('B3 validation regressions',len(tradeoffs),flush=True)
    for r in rows:
        if r['dataset']=='XJTU' and r['step']==.1 and r['parent'] in ('B3','B123'):print(r,flush=True)


if __name__=='__main__':main()
