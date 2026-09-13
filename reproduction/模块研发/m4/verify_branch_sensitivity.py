"""Independently recompute direct-branch scalar sensitivity metrics and gates."""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa,HERE


def main():
    ap=argparse.ArgumentParser();variants=ap.add_mutually_exclusive_group();variants.add_argument('--support-gate',action='store_true');variants.add_argument('--registered',action='store_true');variants.add_argument('--evidence-uniform',action='store_true');variants.add_argument('--support-cv',action='store_true');args=ap.parse_args()
    results=HERE.parent/'results';prefix='m4_support_cv' if args.support_cv else 'm4_evidence_uniform' if args.evidence_uniform else 'm4_registered' if args.registered else 'm4_support_gate' if args.support_gate else 'm4_multiscale'
    root=results/(prefix+'_sensitivity_v1');source=results/(('m4_evidence' if args.evidence_uniform else prefix)+'_screen_v1')
    suffix='_uniform' if args.evidence_uniform else ''
    protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert sa.digest(source/'summary.json')==protocol['source_summary_sha256']
    assert protocol.get('source_child_suffix','')==suffix
    groups=['B'+''.join(s) for n in range(4) for s in itertools.combinations('123',n)]
    with (root/'cells.csv').open() as f:rows=list(csv.DictReader(f))
    saved={(r['cell_id'],r['group']):r for r in rows};assert len(saved)==len(rows)==17520
    buckets=defaultdict(list);error=0.;checked=0
    for ds in ('XJTU','MATR','Tongji'):
        for c in sa.load_cells(ds):
            tail=Path('folds')/(ds+'__'+c.domain)/'predictions'/(c.id+'.npz')
            with np.load(source/tail) as z,np.load(results/'m3_waveform_candidate_v1'/tail) as parent:
                assert np.array_equal(z['y'],parent['y']) and np.array_equal(z['y'],c.y[sa.K:])
                expected={g:z[g] for g in groups}
                for g in groups:
                    np.testing.assert_array_equal(z[g],parent[g])
                    for step in protocol['steps']:
                        expected[g+'4__'+str(step)]=z[g]+step*(z[g+'4'+suffix]-z[g])
                for g,pred in expected.items():
                    e=(np.asarray(pred,float)-np.asarray(z['y'],float))*100
                    values=np.array([np.mean(abs(e)),np.sqrt(np.mean(e*e)),np.quantile(abs(e),.95)])
                    row=saved[c.id,g];assert row['dataset']==ds and row['domain']==c.domain
                    error=max(error,float(np.max(abs(values-[float(row[k]) for k in ('mae','rmse','p95_ae')]))))
                    buckets[ds,g,c.domain].append(values);checked+=1
        print(ds,'all5steps replayed',flush=True)
    assert checked==17520 and error<1e-8
    means=defaultdict(list)
    for (ds,g,_),values in buckets.items():means[ds,g].append(np.mean(values,axis=0))
    table={key:np.mean(values,axis=0) for key,values in means.items()}
    assert len(table)==len(summary['table'])
    for r in summary['table']:assert np.max(abs(table[r['dataset'],r['group']]-[r[k] for k in ('mae','rmse','p95_ae')]))<1e-8
    edges=[]
    for n in range(4):
        for subset in itertools.combinations('1234',n):
            for addition in sorted(set('1234')-set(subset)):
                edges.append(('B'+''.join(sorted((*subset,addition))),'B'+''.join(subset)))
    assert len(edges)==32;passing=[];incremental=[]
    for step in protocol['steps']:
        tag=str(step);group=lambda g:g+'__'+tag if '4' in g else g
        gates={a+'<'+b:bool(all(np.all(table[ds,group(a)]<table[ds,group(b)]) for ds in ('XJTU','MATR','Tongji'))) for a,b in edges}
        assert gates==summary['gates'][tag]
        if all(gates.values()):passing.append(tag)
        if gates['B1234<B123']:incremental.append(tag)
    assert passing==summary['passing_candidates']
    sa.write_json(root/'verification.json',dict(status='PASS',rows=checked,metric_error=error,
        passing_candidates=passing,incremental_passes=incremental,edges_recomputed=160,
        summary_sha256=sa.digest(root/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Independent interpolation metrics,aggregation and gates from audited source caches. No new model fitting or independent generalization.'))
    print('PASS',prefix,'incremental',incremental,flush=True)


if __name__=='__main__':main()
