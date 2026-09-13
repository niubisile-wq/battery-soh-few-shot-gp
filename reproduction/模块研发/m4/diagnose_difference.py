"""Retrospective domain diagnosis, without model fitting or deployment switches."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa,HERE
from evaluate import dump_csv


def main():
    root=HERE.parent/'results/m4_difference_screen_v1'
    audit=json.loads((root/'verification.json').read_text())
    assert audit['status']=='PASS' and audit['summary_sha256']==sa.digest(root/'summary.json')
    frozen=json.loads((root/'frozen_selections.json').read_text())
    original=list(csv.DictReader((root/'cells.csv').open()))
    lookup={(r['cell_id'],r['group']):r for r in original}
    out=HERE.parent/'results/m4_difference_diagnosis_v1';out.mkdir(exist_ok=False)
    rows=[]
    for fold in frozen['selections']:
        ds,domain=fold['fold'].split(':')
        cells=[r['cell_id'] for r in original if r['dataset']==ds and r['domain']==domain and r['group']=='B123']
        for cid in cells:
            path=root/'folds'/fold['fold'].replace(':','__')/'predictions'/(cid+'.npz')
            with np.load(path) as z:
                before=(z['B123']-z['y'])*100
                for target in ('difference','direct'):
                    g='B1234'+('_direct' if target=='direct' else '')
                    after=(z[g]-z['y'])*100;change=after-before
                    rows.append(dict(dataset=ds,domain=domain,cell_id=cid,target=target,
                        gain=fold['choices'][target]['selected']['gain'],
                        parent_bias_pp=float(before.mean()),change_pp=float(change.mean()),
                        direction_improved_fraction=float(np.mean(before*change<0)),
                        **{k+'_delta_pp':float(lookup[cid,g][k])-float(lookup[cid,'B123'][k]) for k in ('mae','rmse','p95_ae')}))
    domains=[]
    for target in ('difference','direct'):
        for ds,domain in sorted({(r['dataset'],r['domain']) for r in rows}):
            rr=[r for r in rows if (r['target'],r['dataset'],r['domain'])==(target,ds,domain)]
            domains.append(dict(target=target,dataset=ds,domain=domain,cells=len(rr),
                **{k:float(np.mean([r[k] for r in rr])) for k in ('gain','parent_bias_pp','change_pp','direction_improved_fraction','mae_delta_pp','rmse_delta_pp','p95_ae_delta_pp')}))
    pairs=[p for p in audit['paired_comparisons'] if p['family']=='target_control']
    dump_csv(out/'cells.csv',rows);dump_csv(out/'domains.csv',domains)
    sa.write_json(out/'summary.json',dict(status='RETROSPECTIVE_DIAGNOSIS_COMPLETE',domains=domains,target_control_intervals=pairs,
        source_sha256=sa.digest(root/'summary.json'),audit_sha256=sa.digest(root/'verification.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Query truth used descriptively only. No deployment gates,oracle selection,new model,or causal proof. Intervals unadjusted for development reuse.'))
    for r in domains:
        if r['dataset']=='XJTU' and r['target']=='difference':print(r,flush=True)
    for p in pairs:
        if p['comparison']=='B1234-B1234_direct':print(p['dataset'],p['metric'],p['delta_pp'],p['ci95_domain_bootstrap_pp'],flush=True)


if __name__=='__main__':main()
