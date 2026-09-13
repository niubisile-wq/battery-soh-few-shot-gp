"""Replay source-only mappings and check nested GP training-key boundaries."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import nested_signal_probe as ns


def audit(out):
    sa=ns.rd.sa
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['status']=='COMPLETE'
    assert sa.digest(out/'source_snapshot/nested_signal_probe.py')==sa.digest(Path(ns.__file__))
    cells=sa.load_cells('MATR');all_rows=[];controls=[];counts=defaultdict(int)
    for fold in protocol['folds']:
        folder=out/'folds'/fold.replace(':','__')
        result=json.loads((folder/'result.json').read_text())
        train,validation,target=sa.split(cells,fold);byid={c.id:c for c in train}
        assert not set(byid)&{c.id for c in validation+target}
        allowed=sa.source_indices(train)
        keys={(c.id,int(j)) for c in train for j in allowed[c.id]}
        assert keys=={tuple(k) for k in result['source_keys']} and len(keys)<=1000
        views={c.id:ns.signal_views(sa.source_view(c,allowed[c.id]),allowed[c.id][allowed[c.id]>=sa.K]) for c in train}
        pairs={}
        for fit in result['fits']:
            excluded=tuple(fit['excluded']);path=out/fit['artifact']
            assert sa.digest(path)==fit['sha256']
            expected={(c.id,int(j)) for c in train if c.domain not in excluded for j in allowed[c.id]}
            assert {tuple(k) for k in fit['fit_keys']}==expected
            model=joblib.load(path)
            visible=[sa.source_view(c,allowed[c.id]) for c in train if c.domain not in excluded]
            z=np.concatenate([model.features.transform(c)[allowed[c.id]] for c in visible])
            np.testing.assert_allclose(z,model.gp.X_train_,rtol=0,atol=1e-12)
            episodes=joblib.load(path.parent/'episodes.joblib');pairs[excluded]=episodes
            parent=fit['group'] if fit['group']!='Physical_control' else 'Base'
            field='base' if fit['group']!='Physical_control' else 'physical'
            for e in episodes[parent]:
                c=byid[e['cell_id']];q=allowed[c.id][allowed[c.id]>=sa.K]
                assert c.domain in excluded and e['query_indices']==q.tolist()
                np.testing.assert_array_equal(e['y'],c.y[q])
                ix=np.r_[np.arange(sa.K),q]
                sparse=sa.inference_view(replace(c,x=c.x[ix],y=c.y[ix],cycle=c.cycle[ix]))
                np.testing.assert_allclose(model.predict(sparse,fit['mode']),e[field],rtol=0,atol=1e-10)
                counts['gp_cell_predictions']+=1
            counts['gp_artifacts']+=1
        original=joblib.load(folder/'source_lodo_episodes.joblib')
        boundaries=json.loads((folder/'source_lodo_audit.json').read_text())
        assert {r['domain'] for r in boundaries}=={c.domain for c in train}
        for boundary in boundaries:
            held=boundary['domain']
            assert {tuple(k) for k in boundary['fit_keys']}=={(c.id,int(j)) for c in train if c.domain!=held for j in allowed[c.id]}
            assert {tuple(k) for k in boundary['held_keys']}=={(c.id,int(j)) for c in train if c.domain==held for j in allowed[c.id]}
        for episodes in original.values():
            assert {e['cell_id'] for e in episodes}==set(byid)
            for e in episodes:
                c=byid[e['cell_id']]
                assert e['domain']==c.domain
                np.testing.assert_array_equal(e['y'],c.y[allowed[c.id][allowed[c.id]>=sa.K]])
        rows={(r['domain'],r['parent'],r['method'],r['cell_id']):r for r in result['rows']}
        assert len(rows)==len(result['rows'])
        for mapping in result['mappings']:
            held=mapping['held'];parent=mapping['parent'];method=mapping['method']
            fit_records=[]
            for domain in sorted({c.domain for c in train}):
                if domain!=held:
                    fit_records.extend(e for e in pairs[tuple(sorted([held,domain]))][parent] if e['domain']==domain)
            val=[e for e in original[parent] if e['domain']==held]
            assert mapping['train_ids']==[e['cell_id'] for e in fit_records]
            assert mapping['validation_ids']==[e['cell_id'] for e in val]
            path=out/mapping['artifact'];assert sa.digest(path)==mapping['sha256']
            predicted,_=ns.learn_predict(fit_records,val,method,views)
            saved=joblib.load(path)
            for e,p in zip(val,predicted,strict=True):
                if 'model' in saved:
                    replay=saved['prior']+saved['model'].predict(saved['scaler'].transform(views[e['cell_id']][method][:,saved['keep']]))
                else: replay=np.full(len(p),saved['prior'])
                np.testing.assert_allclose(p,replay,rtol=0,atol=1e-12)
                a=abs(e['base']-e['y']);b=abs(e['physical']-e['y']);error=np.where(p>0,b,a)
                computed=dict(gain_mae_pp=float(abs(p-(a-b)).mean()*100),selection_mae_pp=float(error.mean()*100),
                    oracle_mae_pp=float(np.minimum(a,b).mean()*100),regret_pp=float((error-np.minimum(a,b)).mean()*100),
                    sign_accuracy=float(np.mean((p>0)==((a-b)>0))))
                r=rows[held,parent,method,e['cell_id']]
                for k,v in computed.items(): np.testing.assert_allclose(r[k],v,rtol=0,atol=1e-10)
                if method=='prior': controls.append(dict(fold=fold,domain=held,parent=parent,parent_mae_pp=float(a.mean()*100),physical_mae_pp=float(b.mean()*100)))
            counts['mapping_replays']+=1
        all_rows.extend(result['rows'])
    summary=json.loads((out/'summary.json').read_text())
    assert sorted(summary['rows'],key=str)==sorted(all_rows,key=str)
    for r in summary['summary']:
        grouped=defaultdict(list)
        for row in all_rows:
            if row['parent']==r['parent'] and row['method']==r['method']: grouped[row['fold'],row['domain']].append(row)
        assert r['cell_episodes']==sum(map(len,grouped.values()))
        for k in ['gain_mae_pp','selection_mae_pp','oracle_mae_pp','regret_pp','sign_accuracy']:
            value=np.mean([np.mean([row[k] for row in group]) for group in grouped.values()])
            np.testing.assert_allclose(r[k],value,rtol=0,atol=1e-10)
    control_summary=[]
    for parent in ['Base','Base+M1']:
        grouped=defaultdict(list)
        for r in controls:
            if r['parent']==parent: grouped[r['fold'],r['domain']].append(r)
        control_summary.append(dict(parent=parent,**{k:float(np.mean([np.mean([r[k] for r in g]) for g in grouped.values()])) for k in ['parent_mae_pp','physical_mae_pp']}))
    result=dict(status='PASS',counts=dict(counts),controls=control_summary,
        limits='Saved GP training inputs and predictions checked; GPs and original outer configuration selection not independently refit.',
        audit_sha256=sa.digest(Path(__file__)),summary_sha256=sa.digest(out/'summary.json'))
    sa.write_json(out/'audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path)
    with threadpool_limits(limits=1): audit(ap.parse_args().out.resolve())
