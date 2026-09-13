"""Cached original-validation selection for additive target innovations."""
import json
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa,HERE
from score import PARENTS


def main():
    results=HERE.parent/'results';source=results/'m4_support_cv_validation_v1'
    audit=json.loads((source/'verification.json').read_text());request=json.loads((source/'request.json').read_text())
    previous=json.loads((source/'result.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(source/'result.json')
    assert audit['request_sha256']==sa.digest(source/'request.json')
    assert all(sa.digest(HERE/n)==v for n,v in request['code_hashes'].items())
    conditional=results/'m4_conditional_mixed_validation_v1'
    ca=json.loads((conditional/'verification.json').read_text())
    assert ca['status']=='PASS' and ca['result_sha256']==sa.digest(conditional/'result.json')
    protocol=json.loads((HERE/'innovation_protocol.json').read_text())
    out=results/'m4_innovation_validation_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'request.json',dict(protocol=protocol,source_audit_sha256=sa.digest(source/'verification.json'),
        source_result_sha256=sa.digest(source/'result.json'),conditional_audit_sha256=sa.digest(conditional/'verification.json'),
        code_hashes={n:sa.digest(HERE/n) for n in ('validate_innovation.py','innovation_protocol.json','innovation_adapter.py')}))
    byds={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')};done=[]
    assert len(previous['selections'])==21
    for entry in previous['selections']:
        fold=entry['fold'];root=Path(entry['root']);assert sa.digest(root/'result.json')==entry['result_sha256']
        old=json.loads((root/'result.json').read_text());assert sa.digest(root/'predictions.joblib')==old['prediction_sha256']
        parent=Path(old['parent_path']);assert sa.digest(parent)==old['parent_sha256']
        biasroot=Path(old['source_root']);assert sa.digest(biasroot/'result.json')==old['source_result_sha256']
        biasresult=json.loads((biasroot/'result.json').read_text())
        assert sa.digest(biasroot/'predictions.joblib')==biasresult['prediction_sha256']
        cached=joblib.load(root/'predictions.joblib')['predictions'];bias=joblib.load(biasroot/'predictions.joblib');vp=joblib.load(parent)
        _,val,_=sa.split(byds[fold.split(':')[0]],fold);rows=[];trials=[];pred={}
        for family in protocol['families']:
            for kernel in protocol['kernels']:
                key=family+'__'+kernel
                for c in val:
                    q=cached[c.id,('uniform' if family=='uniform_innovation' else 'cv')+'__'+kernel]
                    r=bias[c.id,'bias__'+kernel]
                    pred[c.id,key]=q if family=='cv_blend' else q-r
                    for g in PARENTS:
                        for gain in protocol['gains']:
                            p=vp[c.id,g]+gain*(q-vp[c.id,g] if family=='cv_blend' else q-r)
                            rows.append(dict(key=key,target=family,gain=gain,group=g,cell_id=c.id,dataset=c.dataset,
                                domain=c.domain,mae=float(np.mean(abs(p-c.y[sa.K:])))))
                for gain in protocol['gains']:
                    rr=[r for r in rows if r['key']==key and r['gain']==gain and r['group']=='B123']
                    trials.append(dict(target=family,key=key,gain=gain,validation_mae=sa.macro(rr)))
        assert len(trials)==30
        selected={f:min([t for t in trials if t['target']==f],key=lambda t:(t['validation_mae'],t['gain'],t['key'])) for f in protocol['families']}
        control=selected['cv_blend'];prior=old['selected']['cv']
        assert control['key'].replace('cv_blend__','cv__')==prior['key'] and control['gain']==prior['gain']
        assert abs(control['validation_mae']-prior['validation_mae'])<1e-12
        folder=out/'folds'/fold.replace(':','__');folder.mkdir(parents=True)
        joblib.dump(pred,folder/'predictions.joblib',compress=3)
        sa.write_json(folder/'result.json',dict(fold=fold,selected=selected,trials=trials,rows=rows,models=old['models'],
            source_root=str(root),source_result_sha256=sa.digest(root/'result.json'),
            bias_root=str(biasroot),bias_result_sha256=sa.digest(biasroot/'result.json'),
            parent_path=str(parent),parent_sha256=sa.digest(parent),prediction_sha256=sa.digest(folder/'predictions.joblib')))
        done.append(dict(fold=fold,root=str(folder),selected=selected,result_sha256=sa.digest(folder/'result.json')))
        sa.write_json(out/'progress.json',dict(completed=done,total=21));print(fold,selected,flush=True)
    sa.write_json(out/'result.json',dict(status='ALL_VALIDATION_SELECTED_AUDIT_PENDING',selections=done))


if __name__=='__main__':main()
