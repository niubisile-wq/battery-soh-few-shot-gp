"""All21 original validation selections for three frozen-source adapters."""
import json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from conditional_mixed import ConditionalMixed
from score import PARENTS


def main():
    results=HERE.parent/'results';source=results/'m4_mixed_effects_batch_v1/result.json'
    batch=json.loads(source.read_text());assert batch['status']=='VALIDATION_BATCH_COMPLETE' and len(batch['folds'])==21
    protocol=json.loads((HERE/'conditional_mixed_protocol.json').read_text())
    out=results/'m4_conditional_mixed_validation_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'request.json',dict(protocol=protocol,source_batch_sha256=sa.digest(source),
        code_hashes={n:sa.digest(HERE/n) for n in ('validate_conditional_mixed.py','conditional_mixed.py','conditional_mixed_protocol.json')},
        scope='Original validation only,no query scoring or source refitting.'))
    byds={ds:sa.load_cells(ds) for ds in ('XJTU','MATR','Tongji')};done=[]
    with threadpool_limits(limits=1):
        for entry in batch['folds']:
            fold=entry['fold'];root=Path(entry['branch']);req=json.loads((root/'request.json').read_text())
            old=json.loads((root/'result.json').read_text());audit=json.loads((root/'verification.json').read_text())
            assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
            assert audit['request_sha256']==sa.digest(root/'request.json') and audit['code_sha256']==sa.digest(HERE/'audit_mixed_effects.py')
            assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
            parent=Path(req['origin']['validation_parent_path']);assert sa.digest(parent)==req['origin']['validation_parent_sha256']
            vp=joblib.load(parent);oldpredpath=root/'validation_branch_predictions.joblib'
            assert sa.digest(oldpredpath)==old['prediction_sha256'];oldpred=joblib.load(oldpredpath)
            train,val,test=sa.split(byds[fold.split(':')[0]],fold)
            assert set(vp)=={(c.id,g) for c in val for g in PARENTS}
            rows=[];trials=[];pred={};specs=[];boundary=0.;bias_error=0.
            folder=out/'folds'/fold.replace(':','__');folder.mkdir(parents=True)
            for kernel in protocol['kernels']:
                spec=next(m for m in old['models'] if m['key']=='mixed__'+kernel)
                assert sa.digest(Path(spec['path']))==spec['sha256'];model=joblib.load(spec['path']);specs.append(spec)
                assert set(model.source_keys)==set(map(tuple,req['origin']['source_keys']))
                assert {cid for cid,j in model.source_keys}=={c.id for c in train}
                for family in protocol['families']:
                    key=family+'__'+kernel;adapter=model if family=='bias' else ConditionalMixed(model,family=='private')
                    for c in val:
                        q=adapter.predict(sa.inference_view(c));pred[c.id,key]=q
                        if family=='bias':bias_error=max(bias_error,float(np.max(abs(q-oldpred[c.id,'mixed__'+kernel]))))
                        yy=c.y.copy();yy[sa.K:]=999;n=min(len(c.y),sa.K+3)
                        boundary=max(boundary,float(np.max(abs(q-adapter.predict(replace(c,y=yy))))),
                            float(np.max(abs(q[:n-sa.K]-adapter.predict(sa.prefix(c,n))))))
                        for g in PARENTS:
                            for gain in protocol['gains']:
                                p=vp[c.id,g]+gain*(q-vp[c.id,g])
                                rows.append(dict(key=key,target=family,group=g,gain=gain,cell_id=c.id,dataset=c.dataset,
                                    domain=c.domain,mae=float(np.mean(abs(p-c.y[sa.K:])))))
                    for gain in protocol['gains']:
                        rr=[r for r in rows if r['key']==key and r['group']=='B123' and r['gain']==gain]
                        trials.append(dict(target=family,key=key,gain=gain,validation_mae=sa.macro(rr)))
            assert boundary<1e-8 and bias_error<1e-10 and len(trials)==30
            selected={f:min([t for t in trials if t['target']==f],key=lambda t:(t['validation_mae'],t['gain'],t['key'])) for f in protocol['families']}
            assert selected['bias']['gain']==old['selected']['mixed']['gain']
            assert selected['bias']['key'].replace('bias__','mixed__')==old['selected']['mixed']['key']
            joblib.dump(pred,folder/'predictions.joblib',compress=3)
            sa.write_json(folder/'result.json',dict(fold=fold,selected=selected,trials=trials,rows=rows,models=specs,
                parent_path=str(parent),parent_sha256=sa.digest(parent),source_root=str(root),source_result_sha256=sa.digest(root/'result.json'),
                prediction_sha256=sa.digest(folder/'predictions.joblib'),boundary=boundary,bias_error=bias_error))
            done.append(dict(fold=fold,root=str(folder),selected=selected,result_sha256=sa.digest(folder/'result.json')))
            sa.write_json(out/'progress.json',dict(completed=done,total=21));print(fold,'three families selected',flush=True)
    sa.write_json(out/'result.json',dict(status='ALL_VALIDATION_SELECTED_AUDIT_PENDING',selections=done))


if __name__=='__main__':main()
