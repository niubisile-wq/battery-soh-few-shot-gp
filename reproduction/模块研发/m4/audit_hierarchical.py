"""All source model refits, independently based target adaptation and validation."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from mixed_effects_pilot import load_inputs
from hierarchical_branch import fit_branch
from audit_conditional_mixed import independent_predict
from score import PARENTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);root=Path(ap.parse_args().root)
    req=json.loads((root/'request.json').read_text());result=json.loads((root/'result.json').read_text())
    assert result['status']=='VALIDATION_COMPLETE_AUDIT_PENDING'
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    source,val,allowed,vp,refs,origin=load_inputs(req['fold']);assert origin==req['origin']
    assert sa.digest(Path(req['control_root'])/'result.json')==req['control_result_sha256']
    assert sa.digest(root/'predictions.joblib')==result['prediction_sha256'];saved=joblib.load(root/'predictions.joblib')
    keys={f+'__'+k for f in req['protocol']['families'] for k in req['protocol']['kernels']}
    assert len(result['models'])==4 and {m['key'] for m in result['models']}==keys
    errors=dict(replay=0.,refit=0.,independent=0.,row=0.,trial=0.);pred={}
    with threadpool_limits(limits=1):
        for spec in result['models']:
            path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];model=joblib.load(path)
            family,kernel=spec['key'].split('__');fresh=fit_branch(refs[kernel],source,allowed,family=='hierarchical')
            assert model.source_keys==fresh.source_keys
            assert spec['optimization']==model.parent.gp.optimization_ and spec['diagnostics']==model.parent.gp.diagnostics_
            np.testing.assert_allclose(fresh.parent.gp.X_train_,model.parent.gp.X_train_,atol=1e-12,rtol=0)
            for c in val:
                masked=sa.inference_view(c);p=model.predict(masked);pred[c.id,spec['key']]=p
                independent=.5*(independent_predict(model.parent,masked,'shared',point_mean=True)+independent_predict(model.parent,masked,'private',point_mean=True))
                errors['replay']=max(errors['replay'],float(np.max(abs(p-saved[c.id,spec['key']]))))
                errors['refit']=max(errors['refit'],float(np.max(abs(p-fresh.predict(masked)))))
                errors['independent']=max(errors['independent'],float(np.max(abs(p-independent))))
            print(req['fold'],spec['key'],'source refit and independent adaptation',flush=True)
    assert set(pred)==set(saved);byid={c.id:c for c in val}
    expected={(cid,k,g,a) for cid in byid for k in keys for g in PARENTS for a in req['protocol']['gains']}
    assert len(result['rows'])==len(expected) and {(r['cell_id'],r['key'],r['group'],r['gain']) for r in result['rows']}==expected
    values={}
    for row in result['rows']:
        cid,k,g,a=row['cell_id'],row['key'],row['group'],row['gain'];c=byid[cid]
        loss=float(np.mean(abs((1-a)*vp[cid,g]+a*pred[cid,k]-c.y[sa.K:])))
        errors['row']=max(errors['row'],abs(loss-row['mae']))
        if g=='B123':values.setdefault((k,a),{}).setdefault(c.domain,[]).append(loss)
    assert len(result['trials'])==20 and {(t['key'],t['gain']) for t in result['trials']}==set(values)
    trials=[]
    for t in result['trials']:
        loss=float(np.mean([np.mean(v) for v in values[t['key'],t['gain']].values()]))
        errors['trial']=max(errors['trial'],abs(loss-t['validation_mae']));trials.append(dict(t,validation_mae=loss))
    for family in req['protocol']['families']:
        chosen=min([t for t in trials if t['target']==family],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
        assert chosen['key']==result['selected'][family]['key'] and chosen['gain']==result['selected'][family]['gain']
    assert max(errors.values())<1e-8,errors
    sa.write_json(root/'verification.json',dict(status='PASS',errors=errors,models_refitted=4,rows=len(expected),trials=20,
        request_sha256=sa.digest(root/'request.json'),result_sha256=sa.digest(root/'result.json'),code_sha256=sa.digest(Path(__file__)),
        helper_sha256=sa.digest(HERE/'audit_conditional_mixed.py'),
        limits='All4source GP refits use same fitting implementation; target contrast basis independent; audited encoder reused not refit. No held-domain confirmation.'))
    print('PASS',req['fold'],errors,flush=True)


if __name__=='__main__':main()
