"""Independent stored-prediction metric and coverage checks for M2 screens."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import repair_diagnostics as rd
from summarize_strict import recalculate


def audit(root):
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['status']=='COMPLETE'
    cells=[c for d in rd.sa.PROTOCOL['datasets'] for c in rd.sa.load_cells(d)]
    byid={c.id:c for c in cells}
    folds=sorted({f'{c.dataset}:{c.domain}' for c in cells})
    assert protocol['folds']==folds
    data=json.loads((root/'summary.json').read_text())
    seen=set();points=0;max_error=0.;selection_replays=0;validation_replays=0;masked_reference_replays=0;stored_rows=[]
    for fold in folds:
        train,val,test=rd.sa.split(cells,fold)
        allowed=rd.sa.source_indices(train)
        allowed_keys={(c.id,int(j)) for c in train for j in allowed[c.id]}
        for seed in protocol['seeds']:
            path=root/'folds'/fold.replace(':','__')/f'seed_{seed}'
            result=json.loads((path/'results.json').read_text())
            stored_rows.extend(result['rows'])
            grouped=defaultdict(list)
            for r in result['rows']:
                key=(seed,r['group'],r['variant'],r['cell_id'])
                assert key not in seen;seen.add(key)
                assert r['cell_id'] in {c.id for c in test}
                grouped[r['group'],r['cell_id']].append(r)
            for (g,cid),rows in grouped.items():
                with np.load(path/g/(cid+'.npz')) as z:
                    assert np.array_equal(z['y'],byid[cid].y[rd.sa.K:])
                    for r in rows:
                        m=recalculate(z['y'],z[r['variant']]);points+=len(z['y'])
                        for k,v in m.items():
                            if v is None: assert r[k] is None
                            else: max_error=max(max_error,abs(v-r[k]));assert abs(v-r[k])<1e-10
            if (path/'source_lodo_audit.json').exists():
                entries=json.loads((path/'source_lodo_audit.json').read_text())
                union=set()
                for e in entries:
                    fit={tuple(k) for k in e['fit_keys']};held={tuple(k) for k in e['held_keys']}
                    query={tuple(k) for k in e['query_keys']}
                    assert not fit&held and fit|held==allowed_keys
                    assert query=={k for k in held if k[1]>=rd.sa.K}
                    assert all(byid[cid].domain==e['domain'] for cid,j in held)
                    assert all(byid[cid].domain!=e['domain'] for cid,j in fit)
                    assert not union&query;union|=query
                assert union=={k for k in allowed_keys if k[1]>=rd.sa.K}
                episodes=joblib.load(path/'source_lodo_episodes.joblib')
                for parent,group in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
                    ve=episodes[parent];ids={e['cell_id'] for e in ve}
                    assert ids<={c.id for c in train}
                    for e in ve:
                        cid=e['cell_id'];ix=allowed[cid][allowed[cid]>=rd.sa.K]
                        assert np.array_equal(e['y'],byid[cid].y[ix])
                    if (path/(group+'_heads.joblib')).exists():
                        import residual_head
                        fitted=residual_head.fit(ve)
                        saved=joblib.load(path/(group+'_heads.joblib'))
                        for kind in fitted:
                            a,am=fitted[kind];b,bm=saved[kind]
                            np.testing.assert_allclose(a.mean_,b.mean_,atol=1e-12)
                            np.testing.assert_allclose(a.scale_,b.scale_,atol=1e-12)
                            for x,y in zip(am,bm,strict=True):
                                np.testing.assert_allclose(x.coef_,y.coef_,atol=1e-12)
                                np.testing.assert_allclose(x.intercept_,y.intercept_,atol=1e-12)
                        ve=joblib.load(path/(group+'_selection_episodes.joblib'))
                        ids={e['cell_id'] for e in ve}
                        assert ids=={c.id for c in val}
                        for e in ve:
                            assert np.array_equal(e['y'],byid[e['cell_id']].y[rd.sa.K:])
                            original=e['meta_predictions']
                            rebuilt=residual_head.attach(dict(e),saved)['meta_predictions']
                            for kind in original: np.testing.assert_allclose(rebuilt[kind],original[kind],atol=1e-12)
                    if (path/(group+'_reference_heads.joblib')).exists():
                        import reference_gate
                        saved=joblib.load(path/(group+'_reference_heads.joblib'))
                        if saved.get('raw_reference',False): ve=reference_gate.with_raw_reference(ve,episodes['Base'])
                        ref_train=[rd.sa.source_view(c,allowed[c.id]) for c in train]
                        fitted=reference_gate.fit(ve,ref_train,soft='soft_scales' in saved,metric=saved.get('reference_metric',False),bagged='bagged_seeds' in saved,support_gate='support_quantiles' in saved,consensus_gate=saved.get('consensus_gate',False),n_bags=len(saved.get('bagged_seeds',range(24))),risk_objective=saved.get('risk_objective','mae'),raw_reference=saved.get('raw_reference',False),backbone_coverage=saved.get('backbone_coverage',False),common_scale=saved.get('common_scale',False),position_risk=saved.get('position_risk',False),state_risk=saved.get('state_risk',False),state_proxy=saved.get('state_proxy','raw'),physical_cap=saved.get('physical_cap',False))
                        if saved.get('state_risk',False):
                            assert fitted.get('state_proxy','raw')==saved.get('state_proxy','raw')
                            assert fitted['state_width']==saved['state_width']
                            for name in ['state_centers','state_costs']: np.testing.assert_array_equal(fitted[name],saved[name])
                        masked_reference_replays+=1
                        assert fitted.get('risk_sampling')==saved.get('risk_sampling')
                        assert fitted.get('risk_reference_index')==saved.get('risk_reference_index')
                        assert fitted['params']==saved['params']
                        assert fitted.get('risk_objective','mae')==saved.get('risk_objective','mae')
                        if 'bagged_seeds' in saved:
                            assert fitted['bagged_seeds']==saved['bagged_seeds']
                            np.testing.assert_array_equal(fitted['risk_bootstrap_weights'],saved['risk_bootstrap_weights'])
                        assert fitted.get('soft_scales')==saved.get('soft_scales')
                        assert fitted['source_ids']==saved['source_ids']
                        for k in ['costs','weight']: np.testing.assert_allclose(fitted[k],saved[k],atol=1e-12)
                        for view in fitted['views']:
                            a=fitted['views'][view];b=saved['views'][view]
                            for k in ['keep','z']: np.testing.assert_array_equal(a[k],b[k])
                            if 'projection' in a: np.testing.assert_array_equal(a['projection'],b['projection'])
                            if 'support_z' in a:
                                for k in ['support_z','support_radius','support_thresholds']: np.testing.assert_array_equal(a[k],b[k])
                            np.testing.assert_array_equal(a['scaler'].mean_,b['scaler'].mean_)
                            np.testing.assert_array_equal(a['scaler'].scale_,b['scaler'].scale_)
                        ve=joblib.load(path/(group+'_selection_episodes.joblib'))
                        ids={e['cell_id'] for e in ve};assert ids=={c.id for c in val}
                        raw_model=None
                        if saved.get('raw_reference',False):
                            prov=json.loads((rd.OLD/'folds'/fold.replace(':','__')/'provenance.json').read_text())
                            assert {tuple(k) for k in prov['source_keys']}==allowed_keys
                            raw_info=prov['frozen']['Base'];raw_path=rd.sa.ROOT/raw_info['job']/'model.joblib'
                            assert rd.sa.digest(raw_path)==raw_info['model_sha256']
                            raw_model=joblib.load(raw_path)
                        for e in ve:
                            assert np.array_equal(e['y'],byid[e['cell_id']].y[rd.sa.K:])
                            if raw_model is not None:
                                with threadpool_limits(limits=1):
                                    raw,res=rd.sa.predict_components(raw_model,rd.sa.inference_view(byid[e['cell_id']]),raw_info['mode'])
                                np.testing.assert_allclose(e['raw_base'],raw,atol=1e-12,rtol=0.)
                                np.testing.assert_allclose(e['raw_residual'],res,atol=1e-12,rtol=0.)
                            replay=reference_gate.attach(dict(e),byid[e['cell_id']],saved)
                            assert replay['reference_choices']==e['reference_choices']
                            for view in e['reference_predictions']:
                                np.testing.assert_array_equal(replay['reference_predictions'][view],e['reference_predictions'][view])
                    choices=json.loads((path/(group+'_selection.json')).read_text())
                    for v,old in choices.items():
                        replay=rd.select(ve,v,ids,{c.id for c in test})
                        assert replay==old;selection_replays+=1
            else:
                # Rebuild each original validation episode from frozen models;
                # stored summary metrics alone do not verify parameter choice.
                oldjob=rd.OLD/'folds'/fold.replace(':','__')
                prov=json.loads((oldjob/'provenance.json').read_text())
                assert {tuple(k) for k in prov['source_keys']}==allowed_keys
                assert len(allowed_keys)<=1000
                cache={};support={};variance={};posterior={}
                with threadpool_limits(limits=1):
                    for group in rd.sa.JOBS:
                        f=prov['frozen'][group];mp=rd.sa.ROOT/f['job']/'model.joblib'
                        assert rd.sa.digest(mp)==f['model_sha256']
                        model=joblib.load(mp)
                        cache[group]={c.id:rd.sa.predict_components(model,c,f['mode']) for c in val}
                        support[group]={c.id:rd.support_loo_mse(model,c,f['mode']) for c in val}
                        if any(v.startswith('uncertainty') for v in protocol['variants']):
                            variance[group]={c.id:rd.query_variance(model,c,f['mode']) for c in val}
                        if any(v.startswith('posterior') for v in protocol['variants']):
                            posterior[group]={c.id:model.predict(rd.sa.inference_view(c),'posterior') for c in val}
                    for parent,group in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
                        cal=joblib.load(oldjob/f'seed_{seed}'/group/'adapter.joblib')['calibrator']
                        ve=[]
                        for c in val:
                            b,res=cache[parent][c.id]
                            e=dict(cell_id=c.id,dataset=c.dataset,domain=c.domain,y=c.y[rd.sa.K:],
                                   base=b,residual=res,calibrated=cal.predict(b),physical=cache['Physical_control'][c.id][0],
                                   parent_support_mse=support[parent][c.id],physical_support_mse=support['Physical_control'][c.id],
                                   cycles=c.cycle[rd.sa.K:],anchor=float(c.y[rd.sa.K-1]),anchor_cycle=float(c.cycle[rd.sa.K-1]))
                            if variance: e.update(parent_variance=variance[parent][c.id],physical_variance=variance['Physical_control'][c.id])
                            if posterior: e['posterior']=posterior[parent][c.id]
                            ve.append(e)
                        choices=json.loads((path/(group+'_selection.json')).read_text())
                        for v,chosen in choices.items():
                            replay=rd.select(ve,v,{c.id for c in val},{c.id for c in test})
                            assert replay==chosen;validation_replays+=1
    variants=protocol['variants']+['parent','physical_control','strict_old']
    expected={(s,g,v,c.id) for s in protocol['seeds'] for g in ['Base+M2','Base+M1+M2']
              for v in variants for c in cells}
    assert seen==expected
    assert len(data['rows'])==len(seen)
    row_key=lambda r:(r['fold'],r['seed'],r['group'],r['variant'],r['cell_id'])
    assert sorted(stored_rows,key=row_key)==sorted(data['rows'],key=row_key)
    for r in data['summary']:
        rr=[x for x in data['rows'] if all(x[k]==r[k] for k in ['dataset','group','variant'])]
        for k in rd.KEYS:
            dom=defaultdict(list)
            for x in rr: dom[x['domain']].append(x[k])
            value=float(np.mean([np.mean(a) for a in dom.values()]))
            assert abs(value-r[k])<1e-12
    result=dict(status='PASS',scope='Coverage, stored truths, metric recomputation and summary; source LODO selection replay when available',
                folds=len(folds),cells=len(cells),seeds=protocol['seeds'],rows=len(seen),
                prediction_points=points,max_metric_error=max_error,source_lodo_selection_replays=selection_replays,
                validation_selection_replays=validation_replays,
                unbudgeted_source_labels_masked_reference_replays=masked_reference_replays,
                limits='Not a final independent test or a full retraining audit; no efficacy claim is implied.')
    rd.sa.write_json(root/'screen_audit.json',result)
    print(root.name,json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('roots',nargs='+');args=ap.parse_args()
    for name in args.roots: audit(Path(name).resolve())
