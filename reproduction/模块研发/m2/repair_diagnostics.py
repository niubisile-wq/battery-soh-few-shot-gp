"""Fold-local component ablation and conservative joint M2 repair screening.

Reuse the strict source-only OOF calibrators. Select on validation before
scoring any outer query. All results are development evidence, not final tests.
"""
import argparse
from dataclasses import replace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_limits
import strict_ablation as sa

OLD = sa.ROOT/'模块研发/results/m2_strict_ablation_v1'
KEYS = ['mae','rmse','p95_ae']
VARIANTS = ['calibration_only','residual_only','physical_only','correction_only',
            'joint_mean','joint_domain_guard']
INNER_CACHE = None


def candidates(variant):
    alpha = sa.PROTOCOL['m2']['correction_alpha']
    beta = sa.PROTOCOL['m2']['support_beta']
    physical = [0.,.25,.5,.75,1.]
    if variant.startswith('source_lodo_reference_'):
        soft='_soft_' in variant
        view=variant.rsplit('_',1)[1]+('_soft' if soft else '')
        if view in ['pruned','pca4','pca8']: view='hi_'+view
        if '_bagged_' in variant: view+='_bagged'
        supported='_supported_' in variant
        if supported: view+='_supported'
        if '_consensus_' in variant: view+='_consensus'
        indices=range(2,18,3) if '_fixed_' in variant else range(18 if soft or supported else 6)
        return [dict(alpha=0.,beta=0.,physical=0.,reference_view=view,reference_index=i) for i in indices]
    if variant in ['source_lodo_bounded','source_lodo_bounded_variance']:
        return [dict(alpha=0.,beta=b,physical=w,correction_cap=c,
                     variance_cap=int(variant.endswith('variance')))
                for b in [-.5,0.,.5,1.] for w in physical
                for c in ([.25,.5,1.,2.] if variant.endswith('variance') else [.005,.01,.02,.05,.1])]
    if variant in ['temporal_residual','temporal_residual_guard']:
        return [dict(alpha=0.,beta=b,physical=0.,tau=t,gain=g,clip=c)
                for b in [-.5,-.25,0.,.25,.5,.75,1.] for t in [0.,5.,20.,50.,100.]
                for g in [0.,.05,.2] for c in [0.,.05]]
    if variant in ['temporal_only','temporal_fusion']:
        ww=[0.] if variant=='temporal_only' else physical
        return [dict(alpha=0.,beta=0.,physical=w,tau=t,gain=g,clip=c)
                for w in ww for t in [0.,5.,20.,50.,100.] for g in [0.,.05,.2] for c in [0.,.05]]
    if variant in ['posterior_only','posterior_fusion','posterior_guard']:
        ww=[0.] if variant=='posterior_only' else physical
        return [dict(alpha=0.,beta=b,physical=w,posterior_strength=s)
                for b in [-.5,0.,.5,1.] for s in [0.,.25,.5,1.] for w in ww]
    if variant.startswith('source_lodo_learned_'):
        kind='parent' if variant.endswith('parent') else 'dual'
        return [dict(alpha=0.,beta=0.,physical=w,meta_kind=kind,meta_index=i,shrink=s)
                for i in range(5) for s in [0.,.25,.5,1.] for w in physical]
    if variant in ['source_lodo_fixed_extended','source_lodo_residual_extended']:
        ww=physical if variant=='source_lodo_fixed_extended' else [0.]
        return [dict(alpha=0.,beta=b,physical=w) for b in [-1.,-.5,-.25,0.,.25,.5,.75,1.] for w in ww]
    if variant=='source_lodo_fixed':
        return [dict(alpha=0.,beta=b,physical=w) for b in beta for w in physical]
    if variant=='source_lodo_uncertainty':
        return [dict(alpha=0.,beta=b,physical=0.,temperature=t,gate_prior=w,variance_scale=s)
                for b in beta for t in [.25,1.,4.] for w in [.1,.5,.9] for s in [.1,1.,10.]]
    if variant in ['uncertainty_gate','uncertainty_gate_corrected']:
        bb=[0.] if variant=='uncertainty_gate' else beta
        return [dict(alpha=0.,beta=b,physical=0.,temperature=t,gate_prior=w,variance_scale=s)
                for b in bb for t in [.25,1.,4.] for w in [.25,.5,.75] for s in [.1,1.,10.]]
    if variant in ['support_gate','support_gate_corrected']:
        aa=[0.] if variant=='support_gate' else alpha
        bb=[0.] if variant=='support_gate' else beta
        return [dict(alpha=a,beta=b,physical=0.,temperature=t,gate_prior=w)
                for a in aa for b in bb for t in [.25,1.,4.] for w in [.25,.5,.75]]
    if variant=='calibration_only': beta=[0.];physical=[0.]
    if variant=='residual_only': alpha=[0.];physical=[0.]
    if variant=='physical_only': alpha=[0.];beta=[0.]
    if variant=='correction_only': physical=[0.]
    return [dict(alpha=a,beta=b,physical=w) for a in alpha for b in beta for w in physical]


def predict(e, p):
    if 'reference_index' in p: return e['reference_predictions'][p['reference_view']][p['reference_index']]
    corrected=e['base']+p['alpha']*(e['calibrated']-e['base'])+p['beta']*e['residual']
    if 'meta_index' in p:
        corrected=e['base']+p['shrink']*e['meta_predictions'][p['meta_kind']][p['meta_index']]
    if 'posterior_strength' in p:
        corrected=e['base']+p['posterior_strength']*(e['posterior']-e['base'])+p['beta']*e['residual']
    weight=p['physical']
    if 'temperature' in p:
        # Target support LOO scores only. A fixed floor prevents numerical
        # near-zero early-life residuals from creating arbitrarily sharp gates.
        parent_score=e['parent_support_mse']+1e-6
        physical_score=e['physical_support_mse']+1e-6
        if 'variance_scale' in p:
            parent_score=parent_score+p['variance_scale']*e['parent_variance']
            physical_score=physical_score+p['variance_scale']*e['physical_variance']
        logit=(np.log(p['gate_prior']/(1-p['gate_prior']))+
               np.log(parent_score/physical_score)/p['temperature'])
        weight=1/(1+np.exp(-np.clip(logit,-30,30)))
    output=(1-weight)*corrected+weight*e['physical']
    if 'correction_cap' in p:
        cap=p['correction_cap']
        if p['variance_cap']: cap=cap*np.sqrt(e['parent_variance']+1e-6)
        output=corrected+weight*np.clip(e['physical']-corrected,-cap,cap)
    if 'tau' in p:
        from causal_trend import filter_prediction
        output=filter_prediction(output,e['cycles'],e['anchor'],e['anchor_cycle'],p['tau'],p['gain'],p['clip'])
    return output


def support_loo_mse(model,cell,mode):
    """Hold each support label out of anchoring and target conditioning.

    Feature references may use all K observed support signals, but never the
    held support label. GP training is frozen and contains no target cell.
    """
    c=sa.prefix(sa.inference_view(cell),sa.K)
    z=model.features.transform(c)
    raw=model.gp.predict(z)
    if model.mean_model is not None: raw=raw+model.mean_model.predict(z)
    gp=model.gp
    ts=gp.kernel_(gp.X_train_,z)
    cov=gp.kernel_(z)-ts.T@cho_solve((gp.L_,True),ts,check_finite=False)
    cov=(cov+cov.T)/2+np.eye(sa.K)*1e-9
    errors=[]
    for i in range(sa.K):
        keep=np.arange(sa.K)!=i
        shift=float(c.y[keep].mean()) if model.spec.get('anchored',False) else 0.
        p=raw[i]+shift
        residual=c.y[keep]-raw[keep]-shift
        if mode=='source_bias': p+=residual.mean()
        elif mode=='posterior': p+=cov[i,keep]@np.linalg.solve(cov[np.ix_(keep,keep)],residual)
        elif mode!='source_only': raise ValueError(mode)
        errors.append(float(p-c.y[i]))
    return float(np.mean(np.square(errors)))


def query_variance(model,cell,mode):
    c=sa.inference_view(cell);z=model.features.transform(c);gp=model.gp
    _,std=gp.predict(z,return_std=True)
    scale=float(np.asarray(gp._y_train_std).reshape(-1)[0])**2
    variance=std[sa.K:]**2
    if mode=='source_only': return np.maximum(variance,1e-12)
    zs,zq=z[:sa.K],z[sa.K:]
    ts=gp.kernel_(gp.X_train_,zs)
    solved=cho_solve((gp.L_,True),ts,check_finite=False)
    ss=gp.kernel_(zs)-ts.T@solved
    ss=.5*(ss+ss.T)+np.eye(sa.K)*1e-9
    qs=gp.kernel_(zq,zs)-gp.kernel_(zq,gp.X_train_)@solved
    if mode=='posterior':
        variance-=np.sum(qs*np.linalg.solve(ss,qs.T).T,axis=1)*scale
    elif mode=='source_bias': variance+=(ss.mean()-2*qs.mean(1))*scale
    else: raise ValueError(mode)
    return np.maximum(variance,1e-12)


def source_lodo_episodes(train,provenance,target):
    """Pseudo-target entire source domains; never replenish label quotas."""
    allowed=sa.source_indices(train);domains=defaultdict(list)
    if INNER_CACHE is not None:
        root=Path(INNER_CACHE)
        cached=root/'folds'/provenance['fold'].replace(':','__')/'seed_0'
        for name in ['gp.py','features.py','data.py']:
            assert sa.digest(root/'source_snapshot'/name)==sa.digest(sa.M1/name)
        oldprov=json.loads((OLD/'folds'/provenance['fold'].replace(':','__')/'provenance.json').read_text())
        assert oldprov==provenance
        records=json.loads((cached/'source_lodo_audit.json').read_text())
        keys={(c.id,int(j)) for c in train for j in allowed[c.id]}
        for record in records:
            fit={tuple(k) for k in record['fit_keys']};held={tuple(k) for k in record['held_keys']}
            assert not fit&held and fit|held==keys
        result=joblib.load(cached/'source_lodo_episodes.joblib')
        byid={c.id:c for c in train}
        for es in result.values():
            for e in es:
                ix=allowed[e['cell_id']];ix=ix[ix>=sa.K]
                assert np.array_equal(e['y'],byid[e['cell_id']].y[ix])
        for name in ['source_lodo_audit.json','source_lodo_episodes.joblib']:
            shutil.copyfile(cached/name,target/name)
        return result
    for c in train: domains[c.domain].append(c)
    result={'Base':[],'Base+M1':[]};audit=[]
    for domain,held in sorted(domains.items()):
        fit=[c for c in train if c.domain!=domain]
        visible=[sa.source_view(c,allowed[c.id]) for c in fit]
        idx={c.id:allowed[c.id] for c in fit}
        models={}
        for group in sa.JOBS:
            frozen=provenance['frozen'][group]
            models[group]=sa.GPModel(sa.SPECS[frozen['candidate']],frozen['config']).fit(visible,idx)
        audit.append(dict(domain=domain,fit_keys=[[c.id,int(j)] for c in fit for j in idx[c.id]],
                          held_keys=[[c.id,int(j)] for c in held for j in allowed[c.id]],
                          query_keys=[[c.id,int(j)] for c in held for j in allowed[c.id] if j>=sa.K]))
        for cell in held:
            query=allowed[cell.id][allowed[cell.id]>=sa.K]
            if not len(query): continue
            keep=np.r_[np.arange(sa.K),query]
            c=replace(cell,x=cell.x[keep],y=cell.y[keep],cycle=cell.cycle[keep])
            components={};errors={};variance={}
            for group,model in models.items():
                mode=provenance['frozen'][group]['mode']
                components[group]=sa.predict_components(model,c,mode)
                errors[group]=support_loo_mse(model,c,mode)
                variance[group]=query_variance(model,c,mode)
            for parent in result:
                b,r=components[parent]
                result[parent].append(dict(cell_id=c.id,dataset=c.dataset,domain=c.domain,
                    y=c.y[sa.K:],base=b,calibrated=b,residual=r,
                    physical=components['Physical_control'][0],
                    parent_support_mse=errors[parent],physical_support_mse=errors['Physical_control'],
                    parent_variance=variance[parent],physical_variance=variance['Physical_control']))
    sa.write_json(target/'source_lodo_audit.json',audit)
    joblib.dump(result,target/'source_lodo_episodes.joblib',compress=3)
    return result


def select(episodes, variant, validation_ids, test_ids):
    sa.check_validation(episodes,validation_ids,test_ids)
    parent=[sa.metrics(e['y'],e['base']) for e in episodes]
    trials=[]
    for p in candidates(variant):
        rows=sa.score_validation(episodes,[predict(e,p) for e in episodes])
        # Each validation cell represents one source domain in this protocol.
        # Guard all three metrics in every validation domain; parent is feasible.
        violation=max(r[k]-b[k] for r,b in zip(rows,parent,strict=True) for k in KEYS)
        trials.append(dict(p,**{k:sa.macro(rows,k) for k in KEYS},violation=violation))
    feasible=[r for r in trials if variant not in ['joint_domain_guard','source_lodo_learned_guard','posterior_guard','temporal_residual_guard'] or r['violation']<=1e-12]
    if '_minimax_' in variant:
        scale_errors=[sa.metrics(e['y'],e['raw_base']) for e in episodes] if '_common_' in variant else parent
        parent_rows=[dict(b,dataset=e['dataset'],domain=e['domain']) for b,e in zip(scale_errors,episodes,strict=True)]
        scales={k:max(sa.macro(parent_rows,k),1e-8) for k in KEYS}
        for r in trials:
            ratios=[r[k]/scales[k] for k in KEYS]
            r['objective_worst_ratio']=max(ratios);r['objective_mean_ratio']=float(np.mean(ratios))
        pick=min(feasible,key=lambda r:(r['objective_worst_ratio'],r['objective_mean_ratio'],r['mae'],r['rmse'],r['p95_ae']))
    else:
        pick=min(feasible,key=lambda r:(r['mae'],r['rmse'],r['p95_ae'],r['physical'],r['alpha'],abs(r['beta'])))
    return dict(chosen=pick,trials=trials,validation_ids=sorted(validation_ids))


def run(item):
    fold,out,seed=item
    with threadpool_limits(limits=1):
        cells=[c for d in sa.PROTOCOL['datasets'] for c in sa.load_cells(d)]
        tr,va,te=sa.split(cells,fold)
        job=OLD/'folds'/fold.replace(':','__')
        prov=json.loads((job/'provenance.json').read_text())
        assert set(prov['validation_cells'])=={c.id for c in va}
        assert set(prov['test_cells'])=={c.id for c in te}
        allowed={(c.id,int(j)) for c in tr for j in sa.source_indices(tr)[c.id]}
        assert allowed=={tuple(k) for k in prov['source_keys']}
        assert len(allowed)<=1000
        cache={};support={};variances={};posteriors={}
        for group in sa.JOBS:
            frozen=prov['frozen'][group]
            path=sa.ROOT/frozen['job']/'model.joblib'
            assert sa.digest(path)==frozen['model_sha256']
            model=joblib.load(path)
            cache[group]={c.id:sa.predict_components(model,c,frozen['mode']) for c in va+te}
            support[group]={c.id:support_loo_mse(model,c,frozen['mode']) for c in va+te}
            if any(v.startswith('posterior') for v in VARIANTS):
                posteriors[group]={c.id:model.predict(sa.inference_view(c),'posterior') for c in va+te}
            if any(v.startswith(('uncertainty','source_lodo')) for v in VARIANTS):
                variances[group]={c.id:query_variance(model,c,frozen['mode']) for c in va+te}
        rows=[];selections=[]
        target=Path(out)/'folds'/fold.replace(':','__')/f'seed_{seed}'
        target.mkdir(parents=True,exist_ok=True)
        inner=source_lodo_episodes(tr,prov,target) if any(v.startswith('source_lodo') for v in VARIANTS) else None
        for parent,group in [('Base','Base+M2'),('Base+M1','Base+M1+M2')]:
            heads=None
            reference=None
            if any(v.startswith('source_lodo_reference_') for v in VARIANTS):
                import reference_gate
                use_raw=any('_raw_' in v for v in VARIANTS)
                ref_episodes=reference_gate.with_raw_reference(inner[parent],inner['Base']) if use_raw else inner[parent]
                reference=reference_gate.fit(ref_episodes,tr,soft=any('_soft_' in v for v in VARIANTS),
                    metric=any(v.rsplit('_',1)[-1] in ['pruned','pca4','pca8'] for v in VARIANTS),
                    bagged=any('_bagged_' in v for v in VARIANTS),support_gate=any('_supported_' in v for v in VARIANTS),
                    consensus_gate=any('_consensus_' in v for v in VARIANTS),n_bags=128 if any('_b128_' in v for v in VARIANTS) else 24,
                    risk_objective='minimax' if any('_minimax_' in v for v in VARIANTS) else 'mae',raw_reference=use_raw,
                    backbone_coverage=any('_backbone_' in v for v in VARIANTS),common_scale=any('_common_' in v for v in VARIANTS),
                    position_risk=any('_position_' in v for v in VARIANTS),state_risk=any('_state_' in v for v in VARIANTS),
                    state_proxy='physical' if any('_statephys_' in v for v in VARIANTS) else 'raw',
                    physical_cap='raw_supported' if any('_rawsupport_' in v for v in VARIANTS) else any('_physcap_' in v for v in VARIANTS))
                joblib.dump(reference,target/(group+'_reference_heads.joblib'),compress=3)
            if any(v.startswith('source_lodo_learned_') for v in VARIANTS):
                import residual_head
                heads=residual_head.fit(inner[parent])
                joblib.dump(heads,target/(group+'_heads.joblib'),compress=3)
            adapter_path=job/f'seed_{seed}'/group/'adapter.joblib'
            adapter=joblib.load(adapter_path);cal=adapter['calibrator']
            cf=json.loads((adapter_path.parent/'crossfit_audit.json').read_text())
            assert {tuple(k) for k in cf['keys']}=={k for k in allowed if k[1]>=sa.K}
            def episode(c,truth):
                b,r=cache[parent][c.id]
                e=dict(cell_id=c.id,dataset=c.dataset,domain=c.domain,base=b,residual=r,
                       calibrated=cal.predict(b),physical=cache['Physical_control'][c.id][0],
                       parent_support_mse=support[parent][c.id],
                       physical_support_mse=support['Physical_control'][c.id])
                e.update(cycles=c.cycle[sa.K:],anchor=float(c.y[sa.K-1]),anchor_cycle=float(c.cycle[sa.K-1]))
                if variances:
                    e.update(parent_variance=variances[parent][c.id],
                             physical_variance=variances['Physical_control'][c.id])
                if truth: e['y']=c.y[sa.K:]
                if reference is not None and reference.get('raw_reference',False):
                    e.update(raw_base=cache['Base'][c.id][0],raw_residual=cache['Base'][c.id][1])
                if posteriors: e['posterior']=posteriors[parent][c.id]
                if heads is not None: e=residual_head.attach(e,heads)
                if reference is not None: e=reference_gate.attach(e,c,reference)
                return e
            ve=[episode(c,True) for c in va]
            if inner is not None and heads is None and reference is None: ve=inner[parent]
            if heads is not None or reference is not None: joblib.dump(ve,target/(group+'_selection_episodes.joblib'),compress=3)
            # Freeze every candidate before accessing outer query labels.
            choices={v:select(ve,v,{e['cell_id'] for e in ve},{c.id for c in te}) for v in VARIANTS}
            sa.write_json(target/(group+'_selection.json'),choices)
            for v,choice in choices.items():
                selections.append(dict(fold=fold,group=group,seed=seed,variant=v,**choice['chosen']))
            for c in te:
                e=episode(c,False)
                ps={v:predict(e,ch['chosen']) for v,ch in choices.items()}
                ps['parent']=e['base'];ps['physical_control']=e['physical']
                old=adapter['selection']
                ps['strict_old']=sa.combine(e['base'],e['residual'],e['physical'],cal,old)
                # Verify compatibility against immutable stored strict predictions.
                oldrows=json.loads((job/'cell_results.json').read_text())
                ref=next(r for r in oldrows if r['cell_id']==c.id and r['group']==group and r['seed']==seed)
                with np.load(sa.ROOT/ref['prediction_file']) as z:
                    assert np.max(abs(ps['strict_old']-z['pred']))<1e-10
                path=target/group/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,y=c.y[sa.K:],**ps)
                for v,p in ps.items():
                    rows.append(dict(fold=fold,seed=seed,group=group,variant=v,dataset=c.dataset,
                                     domain=c.domain,cell_id=c.id,**sa.metrics(c.y[sa.K:],p)))
        sa.write_json(target/'results.json',dict(rows=rows,selections=selections))
        return dict(fold=fold,seed=seed,rows=rows,selections=selections)


def main():
    global VARIANTS, INNER_CACHE
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--seeds',default='0');ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--family',choices=['components','support','uncertainty','source_lodo','source_extended','learned','posterior','temporal','temporal_residual','source_bounded','reference','reference_soft','reference_metric','reference_bagged','reference_supported','reference_consensus','reference_coverage_stability','reference_minimax','reference_raw','reference_raw_backbone','reference_common_scale','reference_position_risk','reference_state','reference_state_physical','reference_physical_cap','reference_physical_rawcap'],default='components')
    ap.add_argument('--inner-cache')
    args=ap.parse_args();out=Path(args.out).resolve()
    if args.family=='support': VARIANTS=['support_gate','support_gate_corrected']
    if args.family=='uncertainty': VARIANTS=['uncertainty_gate','uncertainty_gate_corrected']
    if args.family=='source_lodo': VARIANTS=['source_lodo_fixed','source_lodo_uncertainty']
    if args.family=='source_extended': VARIANTS=['source_lodo_fixed_extended','source_lodo_residual_extended']
    if args.family=='source_bounded': VARIANTS=['source_lodo_bounded','source_lodo_bounded_variance']
    if args.family=='reference': VARIANTS=['source_lodo_reference_hi','source_lodo_reference_physics']
    if args.family=='reference_soft': VARIANTS=['source_lodo_reference_soft_hi','source_lodo_reference_soft_physics']
    if args.family=='reference_metric': VARIANTS=['source_lodo_reference_pruned','source_lodo_reference_pca4','source_lodo_reference_pca8']
    if args.family=='reference_bagged': VARIANTS=['source_lodo_reference_bagged_pca8']
    if args.family=='reference_supported': VARIANTS=['source_lodo_reference_bagged_supported_pca8']
    if args.family=='reference_consensus': VARIANTS=['source_lodo_reference_bagged_supported_consensus_pca8']
    if args.family=='reference_coverage_stability': VARIANTS=['source_lodo_reference_bagged_supported_b128_pca8','source_lodo_reference_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_minimax': VARIANTS=['source_lodo_reference_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_raw': VARIANTS=['source_lodo_reference_raw_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_raw_backbone': VARIANTS=['source_lodo_reference_raw_backbone_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_common_scale': VARIANTS=['source_lodo_reference_raw_backbone_common_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_position_risk': VARIANTS=['source_lodo_reference_raw_backbone_position_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_state': VARIANTS=['source_lodo_reference_raw_backbone_state_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_state_physical': VARIANTS=['source_lodo_reference_raw_backbone_state_statephys_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_physical_cap': VARIANTS=['source_lodo_reference_raw_backbone_physcap_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='reference_physical_rawcap': VARIANTS=['source_lodo_reference_raw_backbone_physcap_rawsupport_minimax_bagged_supported_fixed_b128_pca8']
    if args.family=='learned': VARIANTS=['source_lodo_learned_parent','source_lodo_learned_dual','source_lodo_learned_guard']
    if args.family=='posterior': VARIANTS=['posterior_only','posterior_fusion','posterior_guard']
    if args.family=='temporal': VARIANTS=['temporal_only','temporal_fusion']
    if args.family=='temporal_residual': VARIANTS=['temporal_residual','temporal_residual_guard']
    INNER_CACHE=str(Path(args.inner_cache).resolve()) if args.inner_cache else None
    cells=[c for d in sa.PROTOCOL['datasets'] for c in sa.load_cells(d)]
    folds=sorted({f'{c.dataset}:{c.domain}' for c in cells});seeds=list(map(int,args.seeds.split(',')))
    protocol=dict(status='RUNNING',purpose='Development component diagnosis and repair screening',
                  variants=VARIANTS,folds=folds,seeds=seeds,source='Reused strict budgeted OOF calibration',
                  code_sha256=sa.digest(Path(__file__)),strict_sha256=sa.digest(Path(sa.__file__)),
                  gate='No increase of MAE/RMSE/P95 in any validation domain for joint_domain_guard',
                  limits='Already explored development data; independent confirmation still required.')
    protocol['candidate_grids']={v:candidates(v) for v in VARIANTS}
    if args.family!='components': protocol['gate']='Target support LOO errors; optional query GP variance; validation-only temperature and prior selection.'
    protocol['source_lodo_cache']=INNER_CACHE
    if args.family in ['reference_physical_cap','reference_physical_rawcap']:
        protocol['physical_cap']='Clip ONLY physical expert at maximum of first K support SOH, in source risk and target inference; fixed rule, no target query labels. Not a universal true-SOH bound.'
        protocol['risk_bootstrap']='128 fixed source-cell resamples, seeds 0..127; coverage quantile .95; raw-backbone coverage and minimax source/validation selection inherited from v8.'
    if args.family=='reference_physical_rawcap':
        protocol['physical_cap']='Physical expert ceiling = max(first K support SOH maximum, current frozen RAW GPR prediction). Same raw reference in both groups; consistent source risk/inference. No future labels or trajectory length; empirical, not guaranteed safe.'
    if args.family in ['source_lodo','source_extended','source_bounded']:
        protocol['gate']='Coefficients selected on held-out source domains, using ONLY the original budgeted labels. Isotonic alpha is fixed zero. Outer validation/query labels never enter M2 coefficient selection.'
    if args.family=='learned':
        protocol['gate']='Residual heads fit to budgeted source-domain-held-out errors; regularization/shrinkage/fusion selected on original validation cells. Outer query labels scoring only.'
    if args.family.startswith('reference'): protocol['gate']='Source OOD risk tables conditioned on first K reference signals; bandwidth, local/global risk mixture and optional soft risk temperature selected on original validation cells.'
    if args.family in ['reference_supported','reference_consensus']:
        protocol['reference_coverage']='Source-only self-excluded nearest-neighbor/local-radius scores; source quantiles .8/.9/.95; novelty shrinks total M2 correction toward parent.'
    if args.family in ['reference_bagged','reference_supported','reference_consensus']:
        protocol['risk_bootstrap']='24 source-cell resamples within source domains, seeds 0..23; GP and reference embedding fixed.'
    if args.family=='reference_consensus':
        protocol['consensus']='Coverage confidence has a floor max(0,2*source-domain improvement fraction-1), only when source mean improvement is positive. No target query labels.'
    if args.family=='reference_coverage_stability':
        protocol['risk_bootstrap']='128 source-cell risk resamples within source domains, fixed seeds 0..127; fixed variant uses source coverage quantile .95 for both parents.'
    if args.family in ['reference_minimax','reference_raw','reference_raw_backbone','reference_common_scale','reference_position_risk','reference_state','reference_state_physical']:
        protocol['risk_bootstrap']='128 source-cell risk resamples, fixed seeds 0..127; fixed source coverage quantile .95, same as v5.'
        protocol['risk_objective']='Minimize worst of MAE/RMSE/P95 ratios to parent risk, tie-break by mean ratio then MAE/RMSE/P95. Applied to source expert routing and fold-local validation hyperparameter selection. Scale floor 1e-8; no outer query truth.'
    if args.family in ['reference_raw','reference_raw_backbone','reference_common_scale','reference_position_risk','reference_state','reference_state_physical']:
        protocol['raw_reference']='Frozen original GPR backbone and support residual available to both groups. Source-OOF expert bank mixes parent and raw GPR at 0/.5/1 before residual/physical correction; identical source backbones collapse duplicates. Same coverage shrinkage toward parent. No new labels or model fitting.'
    if args.family in ['reference_raw_backbone','reference_common_scale','reference_position_risk','reference_state','reference_state_physical']:
        protocol['raw_reference']=protocol['raw_reference'].replace('Same coverage shrinkage toward parent.','Coverage shrinks residual/physical correction toward the risk-selected backbone mixture, not toward the original parent; the backbone mixture itself is not coverage-shrunk.')
    if args.family=='reference_common_scale':
        protocol['risk_objective']=protocol['risk_objective'].replace('ratios to parent risk','ratios to the frozen original GPR risk in both groups')
        protocol['common_scale']='Source expert routing normalizes by the pure original-GPR expert; fold-local validation normalizes by original-GPR validation errors. No target query labels or seed-specific coefficients.'
    if args.family=='reference_position_risk':
        protocol['position_risk']='Compute source risk on linearly interpolated signed error over query record positions, using ONLY original budgeted source query labels. Dense errors are estimates, not observations. Query endpoints must be budgeted. Validation/scoring metrics remain actual full-query metrics; no target query interpolation.'
    if args.family in ['reference_state','reference_state_physical']:
        protocol['state_risk']='Proxy=clip(mean(first K SOH)-current frozen raw GPR prediction,0,1). Source-only domain/cell balanced q20/q50/q80 centers; Gaussian bandwidth=max((q90-q10)/2,1e-3). Per-cell weighted observed-query MAE/RMSE/weighted P95 risks at each center; interpolate routed predictions by current proxy. No full target length, future labels, or interpolated source labels.'
    if args.family=='reference_state_physical':
        protocol['state_risk']=protocol['state_risk'].replace('current frozen raw GPR prediction','current frozen physical-GPR prediction')
    if args.family=='posterior': protocol['gate']='Support K10 posterior update, residual mean and physical fusion coefficients selected on original validation cells.'
    if args.family.startswith('temporal'): protocol['gate']='Causal trend filtering uses last support SOH and past/current predicted SOH and cycles. Filter time scale, trend gain, innovation cap and correction/fusion weights selected on original validation cells.'
    sa.write_json(out/'protocol.json',protocol)
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__),Path(sa.__file__),sa.PROTOCOL_PATH,sa.M1/'gp.py',sa.M1/'features.py',sa.M1/'data.py']:
        shutil.copyfile(path,snapshot/path.name)
    if args.family=='learned': shutil.copyfile(Path(__file__).with_name('residual_head.py'),snapshot/'residual_head.py')
    if args.family.startswith('reference'): shutil.copyfile(Path(__file__).with_name('reference_gate.py'),snapshot/'reference_gate.py')
    if args.family in ['reference_state','reference_state_physical']: shutil.copyfile(Path(__file__).with_name('state_risk.py'),snapshot/'state_risk.py')
    if args.family.startswith('temporal'): shutil.copyfile(Path(__file__).with_name('causal_trend.py'),snapshot/'causal_trend.py')
    rows=[];selections=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(run,(f,str(out),s)) for f in folds for s in seeds]
        for fut in as_completed(fs):
            r=fut.result();rows+=r['rows'];selections+=r['selections']
            print(r['fold'],r['seed'],'complete',flush=True)
    summary=[]
    for d in sa.PROTOCOL['datasets']:
        for g in ['Base+M2','Base+M1+M2']:
            for v in VARIANTS+['parent','physical_control','strict_old']:
                rr=[r for r in rows if r['dataset']==d and r['group']==g and r['variant']==v]
                summary.append(dict(dataset=d,group=g,variant=v,**{k:sa.macro(rr,k) for k in KEYS}))
    sa.write_json(out/'summary.json',dict(summary=summary,rows=rows,selections=selections))
    lines=['# 第二模块拆分与修正筛查','','开发结果；每电芯计算指标，各域等权。单位 SOH 百分点。',
           '','| 数据集 | 父模型 | 变体 | MAE | RMSE | P95 |','|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append('| '+ ' | '.join([r['dataset'],r['group'],r['variant']]+[f'{100*r[k]:.4f}' for k in KEYS])+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    protocol['status']='COMPLETE';sa.write_json(out/'protocol.json',protocol)


if __name__=='__main__': main()
