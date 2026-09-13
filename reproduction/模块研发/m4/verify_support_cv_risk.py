"""Independent cached-risk interpolation, cell metrics, domain means and gates."""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa,HERE


def main():
    results=HERE.parent/'results';root=results/'m4_support_cv_risk_sensitivity_v1'
    source=results/'m4_support_cv_screen_v1';risk=results/'m3_waveform_stability_v2'
    protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert summary['status']=='RISK_SCREEN_COMPLETE_AUDIT_PENDING'
    for path,key in [(source/'summary.json','source_summary_sha256'),(source/'frozen_selections.json','source_selection_sha256'),
                     (risk/'protocol.json','risk_protocol_sha256'),(risk/'summary.json','risk_summary_sha256')]:
        assert sa.digest(path)==protocol[key]
    assert sa.digest(HERE/'support_cv_risk_sensitivity.py')==protocol['code_sha256']
    audit=json.loads((source/'verification.json').read_text())
    assert audit['status']=='PASS' and audit['summary_sha256']==protocol['source_summary_sha256']
    assert protocol['seeds']==list(range(101,111)) and protocol['steps']==[.1,.25,.5,.75,1.]
    choices=json.loads((source/'frozen_selections.json').read_text())['selections']
    gains={s['fold']:s['choices']['cv']['selected']['gain'] for s in choices}
    manifest={r['path']:r['sha256'] for r in summary['source_cache_manifest']}
    assert len(manifest)==3650
    groups=['B'+''.join(s) for n in range(4) for s in itertools.combinations('123',n)]
    with (root/'cells.csv').open() as f:rows=list(csv.DictReader(f))
    saved={(int(r['seed']),r['cell_id'],r['group']):r for r in rows}
    assert len(saved)==len(rows)==175200
    buckets=defaultdict(list);error=0.;checked=0;seen=set()
    for ds in ('XJTU','MATR','Tongji'):
        for c in sa.load_cells(ds):
            fold=ds+':'+c.domain;folder=fold.replace(':','__')
            with np.load(source/'folds'/folder/'predictions'/(c.id+'.npz')) as z:
                y=z['y'].copy();q=z['branch_cv'].copy()
            assert np.array_equal(y,c.y[sa.K:])
            for seed in protocol['seeds']:
                path=risk/'folds'/folder/str(seed)/(c.id+'.npz')
                assert sa.digest(path)==manifest[str(path)];seen.add(str(path))
                with np.load(path) as z:
                    assert np.array_equal(z['y'],y)
                    expected={g:z[g].copy() for g in groups}
                for g in groups:
                    p=expected[g]
                    for step in protocol['steps']:
                        weight=step*gains[fold]
                        expected[g+'4__'+str(step)]=(1-weight)*p+weight*q
                for g,pred in expected.items():
                    e=(np.asarray(pred,float)-np.asarray(y,float))*100
                    values=np.array([np.mean(abs(e)),np.sqrt(np.mean(e*e)),np.quantile(abs(e),.95)])
                    row=saved[seed,c.id,g]
                    assert row['dataset']==ds and row['domain']==c.domain
                    error=max(error,float(np.max(abs(values-[float(row[k]) for k in ('mae','rmse','p95_ae')]))))
                    buckets[seed,ds,g,c.domain].append(values);checked+=1
        print(ds,'all10 draws and5 steps independently replayed',flush=True)
    assert seen==set(manifest) and checked==175200 and error<1e-8
    means=defaultdict(list)
    for (seed,ds,g,_),values in buckets.items():means[seed,ds,g].append(np.mean(values,axis=0))
    table={key:np.mean(values,axis=0) for key,values in means.items()}
    assert len(table)==len(summary['table'])
    table_error=max(float(np.max(abs(table[r['seed'],r['dataset'],r['group']]-[r[k] for k in ('mae','rmse','p95_ae')]))) for r in summary['table'])
    assert table_error<1e-8
    edges=[]
    for n in range(4):
        for subset in itertools.combinations('1234',n):
            for addition in sorted(set('1234')-set(subset)):
                edges.append(('B'+''.join(sorted((*subset,addition))),'B'+''.join(subset)))
    assert len(edges)==32 and len(summary['gates'])==50
    stored={(r['seed'],r['step']):r for r in summary['gates']};counts={};all_pass={}
    for step in protocol['steps']:
        counts[str(step)]={edge:0 for edge in ('B4<B','B1234<B123','B34<B3')};all_pass[str(step)]=0
        group=lambda g:g+'__'+str(step) if '4' in g else g
        for seed in protocol['seeds']:
            gates={a+'<'+b:bool(all(np.all(table[seed,ds,group(a)]<table[seed,ds,group(b)]) for ds in ('XJTU','MATR','Tongji'))) for a,b in edges}
            record=stored[seed,step];assert gates==record['passed']
            for a,b in edges:
                expected={(ds,k):float(table[seed,ds,group(a)][i]-table[seed,ds,group(b)][i])
                          for ds in ('XJTU','MATR','Tongji') for i,k in enumerate(('mae','rmse','p95_ae'))
                          if table[seed,ds,group(a)][i]>=table[seed,ds,group(b)][i]}
                actual={(r['dataset'],r['metric']):r['delta_pp'] for r in record['failures'][a+'<'+b]}
                assert expected.keys()==actual.keys()
                assert all(abs(v-actual[k])<1e-8 for k,v in expected.items())
            for edge in counts[str(step)]:counts[str(step)][edge]+=gates[edge]
            all_pass[str(step)]+=all(gates.values())
    assert counts==summary['pass_counts']
    sa.write_json(root/'verification.json',dict(status='PASS',rows=checked,metric_error=error,table_error=table_error,
        edges_recomputed=1600,pass_counts=counts,all32_pass_counts=all_pass,
        summary_sha256=sa.digest(root/'summary.json'),protocol_sha256=sa.digest(root/'protocol.json'),
        cells_sha256=sa.digest(root/'cells.csv'),code_sha256=sa.digest(Path(__file__)),
        limits='Independent arithmetic/aggregation/gates from fixed audited caches,not fresh GP fitting or independent data confirmation.'))
    print('PASS',counts,'all32',all_pass,flush=True)


if __name__=='__main__':main()

