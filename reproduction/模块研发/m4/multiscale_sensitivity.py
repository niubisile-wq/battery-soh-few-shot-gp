"""Declared uniform global shrinkage of selected direct-branch weights."""
import json
import argparse
from pathlib import Path
import numpy as np
from source_oof import sa,HERE
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def main():
    ap=argparse.ArgumentParser();variants=ap.add_mutually_exclusive_group();variants.add_argument('--support-gate',action='store_true');variants.add_argument('--registered',action='store_true');variants.add_argument('--evidence-uniform',action='store_true');variants.add_argument('--support-cv',action='store_true');args=ap.parse_args()
    prefix='m4_support_cv' if args.support_cv else 'm4_evidence_uniform' if args.evidence_uniform else 'm4_registered' if args.registered else 'm4_support_gate' if args.support_gate else 'm4_multiscale'
    results=HERE.parent/'results';source=results/(('m4_evidence' if args.evidence_uniform else prefix)+'_screen_v1')
    suffix='_uniform' if args.evidence_uniform else ''
    if args.evidence_uniform or args.support_cv:
        audit=json.loads((source/'verification.json').read_text())
        assert audit['status']=='PASS' and audit['summary_sha256']==sa.digest(source/'summary.json')
    out=results/(prefix+'_sensitivity_v1');out.mkdir(exist_ok=False)
    steps=[.1,.25,.5,.75,1.]
    sa.write_json(out/'protocol.json',dict(steps=steps,source_summary_sha256=sa.digest(source/'summary.json'),source_child_suffix=suffix,
        rule='One global scalar multiplies every selected fold gain across all datasets and8parent groups. Report all5 candidates and all32edges.',
        limits='Exploratory sensitivity after observed development screen,not independent confirmation. No new fitting or per-dataset mixing.',
        code_sha256=sa.digest(Path(__file__))))
    rows=[]
    for ds in ('XJTU','MATR','Tongji'):
        for c in sa.load_cells(ds):
            path=source/'folds'/(ds+'__'+c.domain)/'predictions'/(c.id+'.npz')
            with np.load(path) as z:
                y=z['y'];assert np.array_equal(y,c.y[sa.K:])
                for g in PARENTS:
                    rows.append(dict(dataset=ds,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,z[g])))
                    for step in steps:
                        p=(1-step)*z[g]+step*z[g+'4'+suffix]
                        rows.append(dict(dataset=ds,domain=c.domain,cell_id=c.id,group=g+'4__'+str(step),**error_metrics(y,p)))
    assert len(rows)==17520
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={};failures={}
    for step in steps:
        tag=str(step);group=lambda g:g+'__'+tag if '4' in g else g
        failures[tag]={a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=look[ds,group(a)][k]-look[ds,group(b)][k])
            for ds in ('XJTU','MATR','Tongji') for k in ('mae','rmse','p95_ae') if look[ds,group(a)][k]>=look[ds,group(b)][k]] for a,b in addition_edges()}
        gates[tag]={k:not v for k,v in failures[tag].items()}
    passing=[k for k,v in gates.items() if all(v.values())]
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SENSITIVITY_COMPLETE_NOT_ADOPTED',table=table,gates=gates,failures=failures,passing_candidates=passing,rows=len(rows)))
    print(json.dumps(dict(passing_candidates=passing,incremental_passes=[k for k,v in gates.items() if v['B1234<B123']])),flush=True)


if __name__=='__main__':main()
