"""Replay shrink choices, scored predictions, and macro aggregation."""
import argparse
from collections import defaultdict
import json
import importlib.util
from pathlib import Path
import joblib
import numpy as np
import nested_shrink_probe as sp


def audit(out):
    spec=importlib.util.spec_from_file_location('saved_shrink_probe',out/'nested_shrink_probe.py')
    sp=importlib.util.module_from_spec(spec);spec.loader.exec_module(sp)
    sa=sp.ne.ns.rd.sa;result=json.loads((out/'result.json').read_text())
    assert result['status']=='COMPLETE'
    assert Path(sp.__file__).resolve()==out/'nested_shrink_probe.py'
    indexed={(r['fold'],r['domain'],r['parent'],r['method'],r['cell_id']):r for r in result['rows']}
    assert len(indexed)==len(result['rows'])
    seen=set();coefficients=defaultdict(list)
    for record in result['choices']:
        choice=sp.choose(np.array(record['costs']),record['domains'],record['params'])
        assert choice==record['choice']
        assert record['held'] not in record['domains']
        assert not set(record['train_ids'])&set(record['validation_ids'])
        folder=sa.ROOT/'模块研发/results/m2_nested_signal_probe_v1/folds'/record['fold'].replace(':','__')
        original=joblib.load(folder/'source_lodo_episodes.joblib')
        episodes=[e for e in sp.ne.rg.with_raw_reference(original[record['parent']],original['Base']) if e['domain']==record['held']]
        assert [e['cell_id'] for e in episodes]==record['validation_ids']
        for e in episodes:
            for method,p in sp.predictions(e,record['params'],choice).items():
                key=(record['fold'],record['held'],record['parent'],method,e['cell_id'])
                assert key not in seen;seen.add(key)
                m=sa.metrics(e['y'],p)
                for k in sp.METRICS:np.testing.assert_allclose(indexed[key][k],100*m[k],rtol=0,atol=1e-12)
        coefficients[record['parent']].append(choice['coefficient'])
        if 'domain_coefficient' in choice:
            coefficients[record['parent']+'__domain'].append(choice['domain_coefficient'])
    assert seen==set(indexed)
    for summary in result['summary']:
        grouped=defaultdict(list)
        for r in result['rows']:
            if (r['parent'],r['method'])==(summary['parent'],summary['method']):grouped[r['fold'],r['domain']].append(r)
        for k in sp.METRICS:
            value=np.mean([np.mean([r[k] for r in rows]) for rows in grouped.values()])
            np.testing.assert_allclose(summary[k],value,rtol=0,atol=1e-12)
    report=dict(status='PASS',choices=len(result['choices']),metric_rows=len(seen),
        coefficients={k:dict(nonzero=int(np.count_nonzero(v)),total=len(v),values=v) for k,v in coefficients.items()},
        result_sha256=sa.digest(out/'result.json'),audit_sha256=sa.digest(Path(__file__)),
        limits='Replays saved risk choices and validation scores; does not independently refit GPs or reconstruct training risk tensors.')
    sa.write_json(out/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);audit(ap.parse_args().out.resolve())
