"""Refit residual heads with unbudgeted source labels masked, replay scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import nested_signal_residual as sr


def audit(out):
    sa=sr.ns.rd.sa;r=json.loads((out/'result.json').read_text());assert r['status']=='COMPLETE'
    assert sa.digest(out/'nested_signal_residual.py')==sa.digest(Path(sr.__file__))
    cells=sa.load_cells('MATR');root=sa.ROOT/'模块研发/results/m2_nested_signal_probe_v1';cache={}
    rows={(x['fold'],x['domain'],x['parent'],x['method'],x['cell_id']):x for x in r['rows']}
    assert len(rows)==len(r['rows']);seen=set();replays=0
    for record in r['manifest']:
        fold=record['fold']
        if fold not in cache:
            train,_,_=sa.split(cells,fold);allowed=sa.source_indices(train)
            views={c.id:sr.ns.signal_views(sa.source_view(c,allowed[c.id]),allowed[c.id][allowed[c.id]>=sa.K]) for c in train}
            folder=root/'folds'/fold.replace(':','__');source=json.loads((folder/'result.json').read_text());pairs={}
            for excluded in sorted({tuple(f['excluded']) for f in source['fits']}):
                f=next(f for f in source['fits'] if tuple(f['excluded'])==excluded)
                pairs[excluded]=joblib.load((root/f['artifact']).parent/'episodes.joblib')
            original=joblib.load(folder/'source_lodo_episodes.joblib')
            cache[fold]=(train,views,pairs,original)
        train,views,pairs,original=cache[fold];held=record['held'];parent=record['parent'];records=[]
        for domain in sorted({c.domain for c in train}-{held}):
            records.extend(e for e in pairs[tuple(sorted([held,domain]))][parent] if e['domain']==domain)
        validation=[e for e in original[parent] if e['domain']==held]
        assert record['train_ids']==[e['cell_id'] for e in records]
        assert record['validation_ids']==[e['cell_id'] for e in validation]
        assert not set(record['train_ids'])&set(record['validation_ids'])
        path=out/record['artifact'];assert sa.digest(path)==record['sha256'];saved=joblib.load(path)
        head=sr.fit(records,views,record['method']=='anchored_residual');replays+=1
        for e in validation:
            delta=views[e['cell_id']]['delta'];a=sr.correction(delta,head);b=sr.correction(delta,saved)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
            for method,p in [(record['method'],e['base']+a),('unchanged',e['base'])]:
                key=(fold,held,parent,method,e['cell_id']);seen.add(key);m=sa.metrics(e['y'],p)
                for k in ['mae','rmse','p95_ae']:np.testing.assert_allclose(rows[key][k],100*m[k],rtol=0,atol=1e-12)
    assert seen==set(rows)
    for summary in r['summary']:
        grouped=defaultdict(list)
        for row in r['rows']:
            if (row['parent'],row['method'])==(summary['parent'],summary['method']):grouped[row['fold'],row['domain']].append(row)
        for k in ['mae','rmse','p95_ae']:
            value=np.mean([np.mean([x[k] for x in g]) for g in grouped.values()])
            np.testing.assert_allclose(summary[k],value,rtol=0,atol=1e-12)
    report=dict(status='PASS',masked_source_refits=replays,unique_metric_rows=len(rows),
        result_sha256=sa.digest(out/'result.json'),audit_sha256=sa.digest(Path(__file__)),
        limits='Refit residual heads and replayed scoring; inherited GP fits/configuration selection not independently repeated.')
    sa.write_json(out/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path)
    with threadpool_limits(limits=1):audit(ap.parse_args().out.resolve())
