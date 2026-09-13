"""Insert query correction before frozen M2 routing; keep M2 parameters/calibration fixed."""
import joblib
import numpy as np
from screen import sa,SCREEN,GROUPS
import reference_gate as rg


def coefficient(cell,head,selection):
    assert not any(head.get(k,False) for k in ['raw_reference','state_risk','support_gate','consensus_gate'])
    e=dict(base=np.ones(1),residual=0.,physical=np.zeros(1))
    result=rg.attach(e,sa.inference_view(cell),head)
    a=float(result['reference_predictions'][selection['reference_view']][selection['reference_index']][0])
    assert 0<=a<=1
    return a


def combine(parent,branch,w,coeff):
    return {'B3':(1-w)*parent['B']+w*branch,
            'B13':(1-w)*parent['B1']+w*branch,
            'B23':parent['B2']+coeff['B2']*w*(branch-parent['B']),
            'B123':parent['B12']+coeff['B12']*w*(branch-parent['B1'])}


def validation_trials(dest,val,test,entries,weights):
    job=SCREEN/dest.name/'seed_0';heads={};vp={c.id:{} for c in val};co={c.id:{} for c in val}
    for short,group,raw in [('B2','Base+M2','B'),('B12','Base+M1+M2','B1')]:
        head=joblib.load(job/(group+'_reference_heads.joblib'));heads[short]=head
        ee=joblib.load(job/(group+'_selection_episodes.joblib'))
        sa.check_validation(ee,[c.id for c in val],[c.id for c in test]);cached={e['cell_id']:e for e in ee}
        s=entries[group]['selection']
        for c in val:
            e=cached[c.id];vp[c.id][raw]=e['base']
            vp[c.id][short]=e['reference_predictions'][s['reference_view']][s['reference_index']]
            a=coefficient(c,head,s);co[c.id][short]=a
            picks=e['reference_choices'][s['reference_view']][s['reference_index']]
            assert abs(a-np.mean([1-head['params'][j]['physical'] for j in picks]))<1e-12
    trials=[]
    for path in sorted(dest.glob('*.joblib')):
        model=joblib.load(path);branch={c.id:model.predict(sa.inference_view(c)) for c in val}
        for w in weights:
            preds={c.id:combine(vp[c.id],branch[c.id],w,co[c.id]) for c in val}
            for group in GROUPS:
                loss=sa.macro([dict(dataset=c.dataset,domain=c.domain,
                    mae=float(np.mean(abs(preds[c.id][group+'3']-c.y[sa.K:])))) for c in val])
                trials.append(dict(group=group,key=path.stem,weight=w,validation_mae=loss))
    return trials,heads
