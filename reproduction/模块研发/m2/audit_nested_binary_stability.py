"""Replay binary risk votes and source validation scores."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import nested_binary_stability as bs


def audit(out):
    sa=bs.ss.sp.ne.ns.rd.sa;root=sa.ROOT/'模块研发/results'
    r=json.loads((out/'result.json').read_text());assert r['status']=='COMPLETE'
    for p,h in r['input_hashes'].items():assert sa.digest(Path(p))==h
    assert sa.digest(out/'nested_binary_stability.py')==sa.digest(Path(bs.__file__))
    original=json.loads((root/'m2_nested_shrink_probe_v2/result.json').read_text())
    old=json.loads((root/'m2_nested_shrink_sensitivity_v1/result.json').read_text())
    assert bs.decomposition(old['summary'])==r['decomposition']
    choices={(c['fold'],c['held'],c['parent']):c for c in original['choices']}
    rows={(x['seed'],x['fold'],x['domain'],x['parent'],x['cell_id']):x for x in r['rows']}
    assert len(rows)==len(r['rows']);seen=set();cache={}
    for w in r['weights']:
        c=choices[w['fold'],w['held'],w['parent']]
        ix=np.arange(len(c['domains'])) if w['seed']==0 else bs.ss.sample_rows(c['domains'],w['seed'])
        value=bs.binary_weight(np.array(c['costs'])[ix],np.array(c['domains'])[ix],c['params'])
        assert value==w['physical_weight']
        if c['fold'] not in cache:cache[c['fold']]=joblib.load(root/'m2_nested_signal_probe_v1/folds'/c['fold'].replace(':','__')/'source_lodo_episodes.joblib')
        es=cache[c['fold']]
        for e in bs.ss.sp.ne.rg.with_raw_reference(es[c['parent']],es['Base']):
            if e['domain']!=c['held']:continue
            key=(w['seed'],c['fold'],c['held'],c['parent'],e['cell_id'])
            assert key not in seen;seen.add(key)
            m=sa.metrics(e['y'],(1-value)*e['base']+value*e['physical'])
            for k in bs.ss.sp.METRICS:np.testing.assert_allclose(rows[key][k],100*m[k],rtol=0,atol=1e-12)
    assert seen==set(rows) and bs.ss.summarize(r['rows'])==r['summary']
    lookup={(x['seed'],x['parent']):x for x in r['summary']}
    controls={(x['seed'],x['parent'],x['method']):x for x in old['summary']}
    initial={(x['parent'],x['method']):x for x in original['summary']}
    for g in r['gates']:
        differences={}
        for parent in ['Base','Base+M1']:
            reference=controls[g['seed'],parent,g['control']] if g['seed'] else initial[parent,'simple']
            differences[parent]={k:lookup[g['seed'],parent][k]-reference[k] for k in bs.ss.sp.METRICS}
        assert differences==g['differences']
        assert g['strict']==all(v<0 for d in differences.values() for v in d.values())
    report=dict(status='PASS',weight_replays=len(r['weights']),metric_rows=len(rows),
        result_sha256=sa.digest(out/'result.json'),audit_sha256=sa.digest(Path(__file__)),
        limits='Saved training risk tensors replayed; no independent reconstruction of costs or GP refitting.')
    sa.write_json(out/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path)
    with threadpool_limits(limits=1):audit(ap.parse_args().out.resolve())
