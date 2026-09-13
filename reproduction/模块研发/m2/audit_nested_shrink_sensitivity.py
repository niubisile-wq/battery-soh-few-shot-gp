"""Replay paired source risk resampling and every scored prediction."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import nested_shrink_sensitivity as ss


def audit(out):
    sa=ss.sp.ne.ns.rd.sa;root=sa.ROOT/'模块研发/results/m2_nested_shrink_probe_v2'
    protocol=json.loads((out/'protocol.json').read_text());r=json.loads((out/'result.json').read_text())
    assert protocol['status']==r['status']=='COMPLETE'
    assert protocol['input_sha256']==sa.digest(root/'result.json')
    for name,path in [('nested_shrink_probe.py',Path(ss.sp.__file__)),('nested_shrink_sensitivity.py',Path(ss.__file__))]:
        assert sa.digest(out/name)==sa.digest(path)
    original=json.loads((root/'result.json').read_text());choices={(c['fold'],c['held'],c['parent']):c for c in original['choices']}
    rows={(x['seed'],x['fold'],x['domain'],x['parent'],x['method'],x['cell_id']):x for x in r['rows']}
    assert len(rows)==len(r['rows']);seen=set();paired={};cache={}
    for draw in r['draws']:
        c=choices[draw['fold'],draw['held'],draw['parent']]
        ix=ss.sample_rows(c['domains'],draw['seed']);ids=[c['train_ids'][j] for j in ix]
        assert ids==draw['sample_ids'];key=(draw['fold'],draw['held'],draw['seed'])
        if key in paired:assert paired[key]==ids
        paired[key]=ids
        choice=ss.sp.choose(np.array(c['costs'])[ix],np.array(c['domains'])[ix],c['params'])
        assert choice==draw['choice']
        if draw['fold'] not in cache:
            cache[draw['fold']]=joblib.load(sa.ROOT/'模块研发/results/m2_nested_signal_probe_v1/folds'/draw['fold'].replace(':','__')/'source_lodo_episodes.joblib')
        es=cache[draw['fold']]
        for e in ss.sp.ne.rg.with_raw_reference(es[draw['parent']],es['Base']):
            if e['domain']!=draw['held']:continue
            predict=ss.sp.ne.rg.expert_prediction
            simple=predict(e,c['params'][choice['reference']]);complex_=predict(e,c['params'][choice['selected']])
            fixed=predict(e,c['params'][c['choice']['reference']])
            for method,p in dict(simple=simple,fixed_simple=fixed,domain_guarded=simple+choice['domain_coefficient']*(complex_-simple)).items():
                key=(draw['seed'],draw['fold'],draw['held'],draw['parent'],method,e['cell_id'])
                assert key not in seen;seen.add(key)
                m=sa.metrics(e['y'],p)
                for k in ss.sp.METRICS:np.testing.assert_allclose(rows[key][k],100*m[k],rtol=0,atol=1e-12)
    assert seen==set(rows)
    assert ss.summarize(r['rows'])==r['summary']
    lookup={(x['seed'],x['parent'],x['method']):x for x in r['summary']}
    for g in r['gates']:
        diffs={p:{k:lookup[g['seed'],p,'domain_guarded'][k]-lookup[g['seed'],p,g['control']][k] for k in ss.sp.METRICS} for p in ['Base','Base+M1']}
        assert diffs==g['differences']
        assert g['strict_improvement']==all(v<0 for d in diffs.values() for v in d.values())
        assert g['no_regression']==all(v<=1e-10 for d in diffs.values() for v in d.values())
    report=dict(status='PASS',choices=len(r['draws']),paired_samples=len(paired),metric_rows=len(rows),
        result_sha256=sa.digest(out/'result.json'),audit_sha256=sa.digest(Path(__file__)),
        limits='Replayed risk draws, choices and scoring; did not refit GPs or independently reconstruct original training costs.')
    sa.write_json(out/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path)
    with threadpool_limits(limits=1):audit(ap.parse_args().out.resolve())
