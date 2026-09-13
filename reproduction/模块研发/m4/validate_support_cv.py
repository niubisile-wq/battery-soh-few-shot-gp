"""Freeze support_cv/uniform original-validation choices with cached branch controls."""
import json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from support_cv_mixed import SupportCVMixed
from score import PARENTS


def main():
    results=HERE.parent/'results';source=results/'m4_conditional_mixed_validation_v1'
    previous=json.loads((source/'result.json').read_text());audit=json.loads((source/'verification.json').read_text())
    req=json.loads((source/'request.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(source/'result.json')
    assert audit['request_sha256']==sa.digest(source/'request.json')
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    assert len(previous['selections'])==21
    control=results/'m4_evidence_validation_v1'
    control_audit=json.loads((control/'verification.json').read_text())
    assert control_audit['status']=='PASS' and control_audit['result_sha256']==sa.digest(control/'result.json')
    protocol=json.loads((HERE/'support_cv_protocol.json').read_text())
    out=results/'m4_support_cv_validation_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'request.json',dict(protocol=protocol,source_audit_sha256=sa.digest(source/'verification.json'),
        source_result_sha256=sa.digest(source/'result.json'),control_audit_sha256=sa.digest(control/'verification.json'),
        code_hashes={n:sa.digest(HERE/n) for n in ('validate_support_cv.py','support_cv_mixed.py','support_cv_protocol.json','conditional_mixed.py')}))
    byds={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')};done=[]
    with threadpool_limits(limits=1):
        for entry in previous['selections']:
            fold=entry['fold'];root=Path(entry['root']);assert sa.digest(root/'result.json')==entry['result_sha256']
            old=json.loads((root/'result.json').read_text())
            assert sa.digest(root/'predictions.joblib')==old['prediction_sha256']
            parent=Path(old['parent_path']);assert sa.digest(parent)==old['parent_sha256']
            cached=joblib.load(root/'predictions.joblib');vp=joblib.load(parent)
            _,val,_=sa.split(byds[fold.split(':')[0]],fold);rows=[];trials=[];weights={};pred={};boundary=0.
            for spec in old['models']:
                kernel=spec['key'].split('__')[1];assert sa.digest(Path(spec['path']))==spec['sha256']
                model=joblib.load(spec['path']);adapter=SupportCVMixed(model)
                for c in val:
                    w=adapter.weights(sa.inference_view(c));weights[c.id,kernel]=w
                    yy=c.y.copy();yy[sa.K:]=999
                    ww=adapter.weights(replace(c,y=yy));early=adapter.weights(sa.prefix(c,sa.K))
                    boundary=max(boundary,max(abs(w[k]-ww[k]) for k in w),max(abs(w[k]-early[k]) for k in w))
                    for family in protocol['families']:
                        key=family+'__'+kernel;a=w['private'] if family=='cv' else .5
                        q=(1-a)*cached[c.id,'shared__'+kernel]+a*cached[c.id,'private__'+kernel];pred[c.id,key]=q
                        for g in PARENTS:
                            for gain in protocol['gains']:
                                p=vp[c.id,g]+gain*(q-vp[c.id,g])
                                rows.append(dict(key=key,target=family,gain=gain,group=g,cell_id=c.id,dataset=c.dataset,
                                    domain=c.domain,mae=float(np.mean(abs(p-c.y[sa.K:])))))
                for family in protocol['families']:
                    key=family+'__'+kernel
                    for gain in protocol['gains']:
                        rr=[r for r in rows if r['key']==key and r['gain']==gain and r['group']=='B123']
                        trials.append(dict(target=family,key=key,gain=gain,validation_mae=sa.macro(rr)))
            assert len(trials)==20 and boundary<1e-10
            selected={f:min([t for t in trials if t['target']==f],key=lambda t:(t['validation_mae'],t['gain'],t['key'])) for f in protocol['families']}
            control_folder=control/'folds'/fold.replace(':','__')
            control_result=json.loads((control_folder/'result.json').read_text())
            assert selected['uniform']==control_result['selected']['uniform']
            assert sa.digest(control_folder/'predictions.joblib')==control_result['prediction_sha256']
            control_pred=joblib.load(control_folder/'predictions.joblib')['predictions']
            control_error=max(float(np.max(abs(p-control_pred[cid,key]))) for (cid,key),p in pred.items() if key.startswith('uniform__'))
            assert control_error<1e-8
            folder=out/'folds'/fold.replace(':','__');folder.mkdir(parents=True)
            joblib.dump(dict(predictions=pred,weights=weights),folder/'predictions.joblib',compress=3)
            sa.write_json(folder/'result.json',dict(fold=fold,selected=selected,trials=trials,rows=rows,models=old['models'],
                source_root=str(root),source_result_sha256=sa.digest(root/'result.json'),parent_path=str(parent),parent_sha256=sa.digest(parent),
                prediction_sha256=sa.digest(folder/'predictions.joblib'),weight_boundary=boundary,control_error=control_error,
                control_result_sha256=sa.digest(control_folder/'result.json')))
            done.append(dict(fold=fold,root=str(folder),selected=selected,result_sha256=sa.digest(folder/'result.json')))
            sa.write_json(out/'progress.json',dict(completed=done,total=21));print(fold,'support_cv/uniform selected',flush=True)
    sa.write_json(out/'result.json',dict(status='ALL_VALIDATION_SELECTED_AUDIT_PENDING',selections=done))


if __name__=='__main__':main()

