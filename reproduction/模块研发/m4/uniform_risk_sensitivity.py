"""Paired cached M2 source-risk draws, fixed M4 uniform branch and all5 steps."""
import json
from pathlib import Path
import numpy as np
from source_oof import sa,HERE
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def main():
    results=HERE.parent/'results';source=results/'m4_evidence_screen_v1';risk=results/'m3_waveform_stability_v2'
    sourceaudit=json.loads((source/'verification.json').read_text())
    assert sourceaudit['status']=='PASS' and sourceaudit['summary_sha256']==sa.digest(source/'summary.json')
    rp=json.loads((risk/'protocol.json').read_text());assert rp['status']=='COMPLETE'
    assert rp['candidate_sha256']==sa.digest(results/'m3_waveform_candidate_v1/candidate.json')
    frozen=json.loads((source/'frozen_selections.json').read_text());seeds=rp['seeds'];steps=[.1,.25,.5,.75,1.]
    assert seeds==list(range(101,111))
    out=results/'m4_uniform_risk_sensitivity_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'protocol.json',dict(seeds=seeds,steps=steps,source_summary_sha256=sa.digest(source/'summary.json'),
        source_selection_sha256=sa.digest(source/'frozen_selections.json'),risk_protocol_sha256=sa.digest(risk/'protocol.json'),
        risk_summary_sha256=sa.digest(risk/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        fixed='All GP fits,M4uniform predictions and validation selections; global shrinkages same across every dataset/parent.',
        change='Reuse original paired M2 within-source-domain risk resampling101..110. Same parent draw in old/new groups.',
        limits='Exploratory fixed-GP risk sensitivity,not10 full retrainings or independent tests. No source covariance refit,no seed selection; all5steps and10draws reported.'))
    cells={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')};rows=[];manifest=[]
    for s in frozen['selections']:
        fold=s['fold'];ds=fold.split(':')[0];_,_,test=sa.split(cells[ds],fold)
        gain=s['choices']['uniform']['selected']['gain']
        for c in test:
            tail=Path('folds')/fold.replace(':','__')/'predictions'/(c.id+'.npz')
            with np.load(source/tail) as z:q=z['branch_uniform'].copy();y=z['y'].copy()
            assert np.array_equal(y,c.y[sa.K:])
            for seed in seeds:
                path=risk/'folds'/fold.replace(':','__')/str(seed)/(c.id+'.npz')
                with np.load(path) as z:
                    assert np.array_equal(z['y'],y)
                    for g in PARENTS:
                        p=z[g]
                        rows.append(dict(seed=seed,dataset=ds,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,p)))
                        for step in steps:
                            pred=p+step*gain*(q-p)
                            rows.append(dict(seed=seed,dataset=ds,domain=c.domain,cell_id=c.id,group=g+'4__'+str(step),**error_metrics(y,pred)))
                manifest.append(dict(path=str(path),sha256=sa.digest(path)))
        print(fold,'10 risk draws all5steps scored',flush=True)
    assert len(rows)==175200
    table=aggregate(rows,['seed','dataset','group']);lookup={(r['seed'],r['dataset'],r['group']):r for r in table}
    previous=json.loads((risk/'summary.json').read_text())['table']
    parent_error=max(abs(lookup[r['seed'],r['dataset'],r['group']][k]-r[k]) for r in previous for k in ('mae','rmse','p95_ae'))
    assert parent_error<1e-8
    gates=[]
    for seed in seeds:
        for step in steps:
            def group(g):return g+'__'+str(step) if '4' in g else g
            failures={a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=lookup[seed,ds,group(a)][k]-lookup[seed,ds,group(b)][k])
                for ds in cells for k in ('mae','rmse','p95_ae') if lookup[seed,ds,group(a)][k]>=lookup[seed,ds,group(b)][k]] for a,b in addition_edges()}
            gates.append(dict(seed=seed,step=step,passed={k:not v for k,v in failures.items()},failures=failures))
    counts={str(step):{edge:sum(r['passed'][edge] for r in gates if r['step']==step) for edge in ('B4<B','B1234<B123','B34<B3')} for step in steps}
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='RISK_SCREEN_COMPLETE_AUDIT_PENDING',rows=len(rows),table=table,gates=gates,
        pass_counts=counts,parent_metric_error=parent_error,source_cache_manifest=manifest,
        limits='Cached fixed-GP risk sensitivity only, independent replay still pending; no adopted M4.'))
    print(counts,flush=True)


if __name__=='__main__':main()
