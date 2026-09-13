"""Independent target updates and all additive/blend validation objectives."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from audit_support_cv import independent_weight
from audit_conditional_mixed import independent_predict
from score import PARENTS


def main():
    root=HERE.parent/'results/m4_innovation_validation_v1'
    req=json.loads((root/'request.json').read_text());allresults=json.loads((root/'result.json').read_text())
    assert len(allresults['selections'])==21 and allresults['status']=='ALL_VALIDATION_SELECTED_AUDIT_PENDING'
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    source=HERE.parent/'results/m4_support_cv_validation_v1';conditional=HERE.parent/'results/m4_conditional_mixed_validation_v1'
    assert sa.digest(source/'result.json')==req['source_result_sha256']
    assert sa.digest(source/'verification.json')==req['source_audit_sha256']
    assert sa.digest(conditional/'verification.json')==req['conditional_audit_sha256']
    byds={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')}
    errors=dict(prediction=0.,row=0.,trial=0.);total_rows=0
    with threadpool_limits(limits=1):
        for entry in allresults['selections']:
            folder=Path(entry['root']);assert sa.digest(folder/'result.json')==entry['result_sha256']
            r=json.loads((folder/'result.json').read_text());fold=r['fold']
            for name in ('source','bias'):
                assert sa.digest(Path(r[name+'_root'])/'result.json')==r[name+'_result_sha256']
            assert sa.digest(Path(r['parent_path']))==r['parent_sha256']
            assert sa.digest(folder/'predictions.joblib')==r['prediction_sha256']
            saved=joblib.load(folder/'predictions.joblib');vp=joblib.load(r['parent_path'])
            _,val,_=sa.split(byds[fold.split(':')[0]],fold);byid={c.id:c for c in val};pred={}
            for spec in r['models']:
                assert sa.digest(Path(spec['path']))==spec['sha256']
                model=joblib.load(spec['path']);kernel=spec['key'].split('__')[1]
                for c in val:
                    masked=sa.inference_view(c);weight=independent_weight(model,masked)['private']
                    shared=independent_predict(model,masked,'shared',point_mean=True)
                    private=independent_predict(model,masked,'private',point_mean=True)
                    bias=independent_predict(model,masked,'bias',point_mean=True)
                    for family in req['protocol']['families']:
                        w=.5 if family=='uniform_innovation' else weight
                        q=shared+w*(private-shared);p=q if family=='cv_blend' else q-bias
                        key=family+'__'+kernel;pred[c.id,key]=p
                        errors['prediction']=max(errors['prediction'],float(np.max(abs(p-saved[c.id,key]))))
            assert set(pred)==set(saved)
            keys={f+'__'+k for f in req['protocol']['families'] for k in req['protocol']['kernels']}
            expected={(cid,k,g,a) for cid in byid for k in keys for g in PARENTS for a in req['protocol']['gains']}
            assert len(r['rows'])==len(expected) and {(v['cell_id'],v['key'],v['group'],v['gain']) for v in r['rows']}==expected
            values={}
            for row in r['rows']:
                cid,key,g,a=row['cell_id'],row['key'],row['group'],row['gain'];c=byid[cid]
                pp=(1-a)*vp[cid,g]+a*pred[cid,key] if row['target']=='cv_blend' else vp[cid,g]+a*pred[cid,key]
                loss=float(np.mean(abs(pp-c.y[sa.K:])))
                errors['row']=max(errors['row'],abs(loss-row['mae']))
                if g=='B123':values.setdefault((key,a),{}).setdefault(c.domain,[]).append(loss)
            trials=[];assert len(r['trials'])==30 and {(t['key'],t['gain']) for t in r['trials']}==set(values)
            for t in r['trials']:
                loss=float(np.mean([np.mean(v) for v in values[t['key'],t['gain']].values()]))
                errors['trial']=max(errors['trial'],abs(loss-t['validation_mae']));trials.append(dict(t,validation_mae=loss))
            for family in req['protocol']['families']:
                best=min([t for t in trials if t['target']==family],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
                assert best['key']==r['selected'][family]['key'] and best['gain']==r['selected'][family]['gain']
            old=json.loads((Path(r['source_root'])/'result.json').read_text())['selected']['cv']
            control=r['selected']['cv_blend']
            assert control['key'].replace('cv_blend__','cv__')==old['key'] and control['gain']==old['gain']
            assert r['selected']==entry['selected'];total_rows+=len(expected)
            print(fold,'independent innovation validation',flush=True)
    assert max(errors.values())<1e-8,errors
    sa.write_json(root/'verification.json',dict(status='PASS',errors=errors,rows=total_rows,trials=630,folds=21,
        result_sha256=sa.digest(root/'result.json'),request_sha256=sa.digest(root/'request.json'),code_sha256=sa.digest(Path(__file__)),
        helper_hashes={n:sa.digest(HERE/n) for n in ('audit_support_cv.py','audit_conditional_mixed.py')},
        limits='Independent target update/combination arithmetic on saved audited source models;no new source fitting,not independent generalization.'))
    print('PASS630objectives',errors,flush=True)


if __name__=='__main__':main()
