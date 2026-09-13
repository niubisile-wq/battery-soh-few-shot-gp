"""Reference-signal-conditioned source OOD risk table for expert selection."""
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
import strict_ablation as sa
from features import hi,physics

SETTINGS=[(bw,rho) for bw in [.25,1.,4.] for rho in [.5,1.]]
PARAMS=[dict(beta=b,physical=w) for b in [-1.,-.5,-.25,0.,.25,.5,.75,1.] for w in [0.,.25,.5,.75,1.]]


def with_raw_reference(episodes,raw_episodes):
    raw={e['cell_id']:e for e in raw_episodes}
    assert len(raw)==len(raw_episodes)==len(episodes)
    assert set(raw)=={e['cell_id'] for e in episodes}
    result=[]
    for e in episodes:
        r=raw[e['cell_id']]
        assert e['domain']==r['domain'] and np.array_equal(e['y'],r['y'])
        assert e['base'].shape==r['base'].shape
        result.append(dict(e,raw_base=r['base'],raw_residual=r['residual']))
    return result


def expert_prediction(e,p):
    mix=p.get('raw_mix',0.)
    base=e['base'];residual=e['residual']
    if mix:
        base=(1-mix)*base+mix*e['raw_base']
        residual=(1-mix)*residual+mix*e['raw_residual']
    return (1-p['physical'])*(base+p['beta']*residual)+p['physical']*e['physical']


def expert_risk(e,p,query_indices=None,n_query=None):
    pred=expert_prediction(e,p)
    if query_indices is None:
        m=sa.metrics(e['y'],pred)
    else:
        # Only budgeted error observations; dense values are interpolation,
        # not extra labels. Positions match the per-record evaluation measure.
        q=np.asarray(query_indices,dtype=int)
        if len(q)!=len(pred) or len(q)<2 or np.any(np.diff(q)<=0): raise ValueError('Invalid risk sample positions')
        if q[0]!=sa.K or q[-1]!=sa.K+n_query-1: raise ValueError('Risk samples must span the query grid')
        errors=np.interp(np.arange(sa.K,sa.K+n_query),q,pred-e['y'])
        m=sa.metrics(np.zeros(n_query),errors)
    return [m[k] for k in ['mae','rmse','p95_ae']]


def risk_indices(risks,params,objective='mae',reference_index=None):
    """Rank experts on source risks; preserve the legacy lexicographic default."""
    keys=(risks[...,2],risks[...,1],risks[...,0])
    if objective=='minimax':
        parent=reference_index if reference_index is not None else next(j for j,p in enumerate(params) if p['beta']==0 and p['physical']==0)
        ratios=risks/np.maximum(risks[...,parent:parent+1,:],1e-8)
        keys=keys+(ratios.mean(-1),ratios.max(-1))
    elif objective!='mae':
        raise ValueError('Unknown reference risk objective: '+objective)
    return np.lexsort(keys,axis=-1)[...,0]


def bootstrap_weights(head):
    domains=head['source_domains'];sizes=head['domain_sizes'];draws=[]
    for seed in head['bagged_seeds']:
        rng=np.random.default_rng(seed);weight=np.zeros(len(domains))
        for domain in sorted(sizes):
            idx=np.flatnonzero(np.asarray(domains)==domain)
            probability=head['weight'][idx];probability=probability/probability.sum()
            selected=rng.choice(idx,sizes[domain],replace=True,p=probability)
            weight+=np.bincount(selected,minlength=len(domains))/sizes[domain]/len(sizes)
        draws.append(weight)
    return np.stack(draws)


def descriptor(cell,view):
    c=sa.prefix(sa.inference_view(cell),sa.K)
    x=hi(c.x) if view.startswith('hi') else physics(c)
    return np.r_[x.mean(0),x.std(0)]


def physical_envelope(e,cell,mode=True):
    ceiling=float(np.max(cell.y[:sa.K]))
    if mode=='raw_supported':
        if 'raw_base' not in e: raise ValueError('Raw-supported cap requires raw GPR predictions')
        ceiling=np.maximum(ceiling,e['raw_base'])
    elif mode is not True:
        raise ValueError('Unknown physical cap mode')
    return dict(e,physical=np.minimum(e['physical'],ceiling))


def fit(episodes,cells,soft=False,metric=False,bagged=False,support_gate=False,consensus_gate=False,n_bags=24,risk_objective='mae',raw_reference=False,backbone_coverage=False,common_scale=False,position_risk=False,state_risk=False,state_proxy='raw',physical_cap=False):
    byid={c.id:c for c in cells};counts=Counter(e['domain'] for e in episodes)
    if physical_cap: episodes=[physical_envelope(e,byid[e['cell_id']],physical_cap) for e in episodes]
    weight=np.asarray([1/counts[e['domain']] for e in episodes]);weight/=weight.sum()
    params=PARAMS
    if raw_reference:
        if soft or consensus_gate: raise ValueError('Raw reference currently supports hard/bagged routing only')
        same=all(np.array_equal(e['base'],e['raw_base']) and np.array_equal(e['residual'],e['raw_residual']) for e in episodes)
        # Identical backbones give identical experts: do not triple their mass.
        params=[dict(p,raw_mix=m) for m in ([0.] if same else [0.,.5,1.]) for p in PARAMS]
    costs=[]
    sampling={};allowed=sa.source_indices(cells) if position_risk else None
    for e in episodes:
        q=None;n=None
        if position_risk:
            c=byid[e['cell_id']];q=allowed[c.id][allowed[c.id]>=sa.K];n=len(c.x)-sa.K
            assert np.array_equal(e['y'],c.y[q])
            sampling[c.id]=dict(query_indices=q.tolist(),n_query=n)
        costs.append([expert_risk(e,p,q,n) for p in params])
    costs=np.asarray(costs);views={}
    for view in ['hi','physics']:
        x=np.stack([descriptor(byid[e['cell_id']],view) for e in episodes])
        # Drop constant reference dimensions before comparing regimes.
        keep=x.std(0)>1e-9
        if not keep.any(): keep[:]=True
        scaler=StandardScaler().fit(x[:,keep],sample_weight=weight)
        views[view]=dict(keep=keep,scaler=scaler,z=scaler.transform(x[:,keep]))
    if metric:
        x=np.stack([descriptor(byid[e['cell_id']],'hi') for e in episodes])
        floors=np.tile(np.r_[np.full(7,1e-6),np.full(7,1e-4),np.full(7,1e-3)],2)
        eligible=np.flatnonzero(x.std(0)>floors);selected=[]
        for j in eligible:
            if not selected or np.max(abs(np.corrcoef(x[:,selected+[j]].T)[-1,:-1]))<.999:
                selected.append(int(j))
        if not selected: raise ValueError('No resolved reference dimensions')
        keep=np.zeros(x.shape[1],dtype=bool);keep[selected]=True
        scaler=StandardScaler().fit(x[:,keep],sample_weight=weight)
        z=scaler.transform(x[:,keep])
        views['hi_pruned']=dict(keep=keep,scaler=scaler,z=z)
        _,_,vt=np.linalg.svd(z*np.sqrt(weight[:,None]),full_matrices=False)
        for n in [4,8]:
            projection=vt[:min(n,len(episodes)-1,z.shape[1])].T
            views[f'hi_pca{n}']=dict(keep=keep,scaler=scaler,z=z@projection,projection=projection)
    head=dict(views=views,weight=weight,costs=costs,source_ids=[e['cell_id'] for e in episodes],params=params,settings=SETTINGS)
    if physical_cap: head['physical_cap']=physical_cap
    if raw_reference: head['raw_reference']=True
    if position_risk: head.update(position_risk=True,risk_sampling=sampling)
    if common_scale:
        if not raw_reference or risk_objective!='minimax': raise ValueError('Common scale requires raw minimax routing')
        mix=1. if any(p['raw_mix']==1. for p in params) else 0.
        head['common_scale']=True
        head['risk_reference_index']=next(j for j,p in enumerate(params) if p['beta']==0 and p['physical']==0 and p['raw_mix']==mix)
    if backbone_coverage:
        if not raw_reference or not bagged or not support_gate: raise ValueError('Backbone coverage requires raw bagged supported routing')
        head['backbone_coverage']=True
    if risk_objective!='mae':
        if risk_objective!='minimax' or soft: raise ValueError('Unsupported risk objective configuration')
        head['risk_objective']=risk_objective
    if soft: head['soft_scales']=[.05,.2,1.]
    if metric: head['reference_metric']=True
    if bagged:
        head['source_domains']=[e['domain'] for e in episodes]
        head['domain_sizes']=dict(counts);head['bagged_seeds']=list(range(n_bags))
        head['risk_bootstrap_weights']=bootstrap_weights(head)
        head['views']={'hi_pca8':head['views']['hi_pca8']}
    if support_gate:
        head['support_quantiles']=[.8,.9,.95]
        for model in head['views'].values():
            z=model['z'];d=((z[:,None]-z[None,:])**2).mean(2)
            if len(z)<2: raise ValueError('Coverage estimation requires at least two source references')
            np.fill_diagonal(d,np.inf)
            radius=np.median(np.sort(d,axis=1)[:,:min(3,len(z)-1)],axis=1)
            radius=np.maximum(radius,1e-8)
            near=d.argmin(1);scores=d[np.arange(len(z)),near]/radius[near]
            order=np.argsort(scores);cdf=np.cumsum(weight[order])
            thresholds=np.maximum(np.interp(head['support_quantiles'],cdf,scores[order]),1e-8)
            model.update(support_z=z.copy(),support_radius=radius,support_thresholds=thresholds)
    if consensus_gate: head['consensus_gate']=True
    if state_risk:
        if not raw_reference or position_risk: raise ValueError('State risks require raw reference and observed source errors')
        import state_risk as sr
        head['state_proxy']=state_proxy
        head=sr.fit(episodes,cells,head)
    return head


def attach(e,cell,head):
    original_physical=e['physical']
    if head.get('physical_cap',False): e=physical_envelope(e,cell,head['physical_cap'])
    if head.get('state_risk',False):
        import state_risk as sr
        result=sr.attach(e,cell,head)
        result['physical']=original_physical
        return result
    outputs={};choices={}
    for view,model in head['views'].items():
        z=model['scaler'].transform(descriptor(cell,view)[model['keep']][None])[0]
        if 'projection' in model: z=z@model['projection']
        distance=np.mean((model['z']-z)**2,axis=1)
        preds=[];picks=[];soft_preds=[];soft_picks=[];bag_preds=[];bag_picks=[]
        params=head.get('params',PARAMS)
        # Pure physical predictions repeat across beta values. Do not give
        # duplicate predictions an artificial prior advantage in soft mixing.
        unique={}
        for j,p in enumerate(params): unique.setdefault((1-p['physical'],(1-p['physical'])*p['beta']),j)
        indices=list(unique.values())
        for bw,rho in head.get('settings',SETTINGS):
            logw=-distance/(2*bw*bw)+np.log(head['weight'])
            local=np.exp(logw-logw.max());local/=local.sum()
            weights=(1-rho)*head['weight']+rho*local
            risks=np.einsum('i,ijk->jk',weights,head['costs'])
            index=int(risk_indices(risks,params,head.get('risk_objective','mae'),head.get('risk_reference_index')))
            p=params[index]
            preds.append(expert_prediction(e,p))
            picks.append(index)
            if 'risk_bootstrap_weights' in head:
                bw0=head['risk_bootstrap_weights']
                logb=-distance[None,:]/(2*bw*bw)+np.where(bw0>0,np.log(np.maximum(bw0,1e-300)),-np.inf)
                localb=np.exp(logb-logb.max(1,keepdims=True));localb/=localb.sum(1,keepdims=True)
                wb=(1-rho)*bw0+rho*localb
                rb=np.einsum('bi,ijk->bjk',wb,head['costs'])
                ii=risk_indices(rb,params,head.get('risk_objective','mae'),head.get('risk_reference_index'))
                ps=[params[j] for j in ii]
                a=float(np.mean([1-p['physical'] for p in ps]))
                b=float(np.mean([(1-p['physical'])*p['beta'] for p in ps]))
                w=float(np.mean([p['physical'] for p in ps]))
                pred=a*e['base']+b*e['residual']+w*e['physical']
                if head.get('raw_reference',False):
                    c=float(np.mean([(1-p['physical'])*p['raw_mix'] for p in ps]))
                    d=float(np.mean([(1-p['physical'])*p['raw_mix']*p['beta'] for p in ps]))
                    pred=pred+c*(e['raw_base']-e['base'])+d*(e['raw_residual']-e['residual'])
                bag_preds.append(pred);bag_picks.append(ii.tolist())
            for scale in head.get('soft_scales',[]):
                score=risks[indices,0]
                temperature=scale*max(float(score.std()),1e-4)
                logp=-(score-score.min())/temperature
                probability=np.exp(logp);probability/=probability.sum()
                pp=[params[j] for j in indices]
                a=sum(q*(1-p['physical']) for q,p in zip(probability,pp,strict=True))
                b=sum(q*(1-p['physical'])*p['beta'] for q,p in zip(probability,pp,strict=True))
                w=sum(q*p['physical'] for q,p in zip(probability,pp,strict=True))
                soft_preds.append(a*e['base']+b*e['residual']+w*e['physical'])
                soft_picks.append(dict(parent=a,residual=b,physical=w))
        outputs[view]=np.stack(preds);choices[view]=picks
        if soft_preds:
            outputs[view+'_soft']=np.stack(soft_preds);choices[view+'_soft']=soft_picks
        if bag_preds:
            outputs[view+'_bagged']=np.stack(bag_preds);choices[view+'_bagged']=bag_picks
            if 'support_quantiles' in head:
                dist=((model['support_z']-z)**2).mean(1);nearest=int(dist.argmin())
                score=dist[nearest]/model['support_radius'][nearest]
                bounded=[];support_picks=[]
                consensus_preds=[];consensus_picks=[]
                for i,pred in enumerate(bag_preds):
                    floor=0.
                    coverage_base=e['base']
                    if head.get('backbone_coverage',False):
                        raw_mix=float(np.mean([params[j]['raw_mix'] for j in bag_picks[i]]))
                        coverage_base=e['base']+raw_mix*(e['raw_base']-e['base'])
                    if head.get('consensus_gate',False):
                        parent_index=next(j for j,p in enumerate(params) if p['beta']==0 and p['physical']==0)
                        gain=head['costs'][:,parent_index,0]-head['costs'][:,bag_picks[i],0].mean(1)
                        domains=np.asarray(head['source_domains']);benefits=[]
                        for domain in sorted(set(domains)):
                            mask=domains==domain;w=head['weight'][mask]
                            benefits.append(float(np.dot(w,gain[mask])/w.sum()))
                        if np.mean(benefits)>0:
                            floor=max(0.,2*np.mean(np.asarray(benefits)>1e-12)-1.)
                    for threshold in model['support_thresholds']:
                        confidence=min(1.,float(threshold)/max(float(score),1e-8))
                        bounded.append(coverage_base+confidence*(pred-coverage_base))
                        support_picks.append(dict(bagged_index=i,reference_novelty=float(score),confidence=confidence))
                        if head.get('consensus_gate',False):
                            confidence=max(confidence,float(floor))
                            consensus_preds.append(e['base']+confidence*(pred-e['base']))
                            consensus_picks.append(dict(bagged_index=i,reference_novelty=float(score),confidence=confidence,source_consensus_floor=float(floor)))
                outputs[view+'_bagged_supported']=np.stack(bounded)
                choices[view+'_bagged_supported']=support_picks
                if consensus_preds:
                    outputs[view+'_bagged_supported_consensus']=np.stack(consensus_preds)
                    choices[view+'_bagged_supported_consensus']=consensus_picks
    e['reference_predictions']=outputs;e['reference_choices']=choices
    e['physical']=original_physical
    return e
