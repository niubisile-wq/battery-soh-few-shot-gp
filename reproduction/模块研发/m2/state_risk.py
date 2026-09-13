"""Budgeted source risk conditioned on a causal predicted-degradation proxy."""
import numpy as np


def proxy(raw_prediction,support_labels):
    return np.clip(float(np.mean(support_labels))-np.asarray(raw_prediction),0.,1.)


def observed_state(e,cell,kind='raw'):
    import reference_gate as rg
    if kind not in ['raw','physical']: raise ValueError('Unknown causal state proxy')
    return proxy(e['raw_base' if kind=='raw' else 'physical'],cell.y[:rg.sa.K])


def weighted_quantile(values,weights,q):
    order=np.argsort(values);x=np.asarray(values)[order];w=np.asarray(weights)[order]
    keep=w>0;x=x[keep];w=w[keep]
    if not len(x): raise ValueError('Empty weighted sample')
    if len(x)==1: return float(x[0])
    positions=np.cumsum(w)-.5*w
    positions=(positions-positions[0])/(positions[-1]-positions[0])
    return float(np.interp(q,positions,x))


def fit(episodes,cells,head):
    import reference_gate as rg
    byid={c.id:c for c in cells}
    states=[observed_state(e,byid[e['cell_id']],head.get('state_proxy','raw')) for e in episodes]
    values=np.concatenate(states)
    weights=np.concatenate([np.full(len(s),w/len(s)) for s,w in zip(states,head['weight'],strict=True)])
    centers=np.unique([weighted_quantile(values,weights,q) for q in [.2,.5,.8]])
    width=max((weighted_quantile(values,weights,.9)-weighted_quantile(values,weights,.1))/2,1e-3)
    costs=[]
    for e,state in zip(episodes,states,strict=True):
        errors=np.stack([rg.expert_prediction(e,p)-e['y'] for p in head['params']])
        absolute=abs(errors);per_center=[]
        for center in centers:
            logw=-.5*((state-center)/width)**2
            w=np.exp(logw-logw.max());w/=w.sum()
            per_center.append(np.stack([absolute@w,np.sqrt(errors**2@w),
                [weighted_quantile(a,w,.95) for a in absolute]],axis=1))
        costs.append(per_center)
    head.update(state_risk=True,state_centers=centers,state_width=width,state_costs=np.asarray(costs))
    return head


def attach(e,cell,head):
    import reference_gate as rg
    centers=head['state_centers'];state=observed_state(e,cell,head.get('state_proxy','raw'))
    basis=np.eye(len(centers))
    interpolation=np.stack([np.interp(state,centers,row) for row in basis])
    outputs=[]
    for i in range(len(centers)):
        sub=dict(head,state_risk=False,costs=head['state_costs'][:,i])
        outputs.append(rg.attach(dict(e),cell,sub))
    predictions={};choices={}
    for view in outputs[0]['reference_predictions']:
        predictions[view]=sum(w[None,:]*r['reference_predictions'][view]
                              for w,r in zip(interpolation,outputs,strict=True))
        route='current_raw_prediction_gap' if head.get('state_proxy','raw')=='raw' else 'current_physical_prediction_gap'
        choices[view]=[dict(state_centers=centers.tolist(),routing=route,
                            center_choices=[r['reference_choices'][view][i] for r in outputs])
                       for i in range(len(predictions[view]))]
    e.update(reference_predictions=predictions,reference_choices=choices)
    return e
