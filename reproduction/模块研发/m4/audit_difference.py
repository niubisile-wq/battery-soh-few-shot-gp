"""Refit every difference/direct GP and independently recompute validation rows."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from difference_branch import DifferenceBranch
from score import PARENTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);root=Path(ap.parse_args().root)
    req=json.loads((root/'request.json').read_text());result=json.loads((root/'result.json').read_text())
    assert result['status']=='VALIDATION_COMPLETE_AUDIT_PENDING'
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    provpath=HERE.parent/'results/m2_strict_ablation_v1/folds'/req['fold'].replace(':','__')/'provenance.json'
    assert sa.digest(provpath)==req['provenance_sha256']
    assert json.loads(provpath.read_text())['source_keys']==req['source_keys']
    parentpath=Path(req['validation_parent_path']);assert sa.digest(parentpath)==req['validation_parent_sha256']
    predpath=root/'validation_branch_predictions.joblib';assert sa.digest(predpath)==result['prediction_sha256']
    vp,saved=joblib.load(parentpath),joblib.load(predpath)
    train,val,test=sa.split(sa.load_cells(req['fold'].split(':')[0]),req['fold'])
    allowed={}
    for cid,j in req['source_keys']:allowed.setdefault(cid,[]).append(j)
    assert set(allowed)=={c.id for c in train} and set(allowed).isdisjoint(c.id for c in val+test)
    assert sum(map(len,allowed.values()))==len(set(map(tuple,req['source_keys'])))<=1000
    visible=[sa.source_view(c,allowed[c.id]) for c in train]
    keys={t+'__'+k for t in req['protocol']['targets'] for k in req['protocol']['kernels']}
    assert len(result['models'])==4 and {m['key'] for m in result['models']}==keys
    predictions={};error=0.;refit_error=0.;latent={}
    with threadpool_limits(limits=1):
        for spec in result['models']:
            path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];model=joblib.load(path)
            target,kernel=spec['key'].split('__')
            assert model.target==target and model.kernel==kernel
            assert set(model.source_keys)==set(map(tuple,req['source_keys']))
            assert len(model.gp.X_train_)==len(model.source_keys)
            fresh=DifferenceBranch(target,kernel).fit(visible,allowed)
            assert fresh.source_keys==model.source_keys
            np.testing.assert_allclose(fresh.gp.X_train_,model.gp.X_train_,atol=1e-12,rtol=0)
            latent[spec['key']]=model.gp.X_train_
            for c in val:
                p=model.predict(sa.inference_view(c));predictions[c.id,spec['key']]=p
                error=max(error,float(np.max(abs(p-saved[c.id,spec['key']]))))
                refit_error=max(refit_error,float(np.max(abs(p-fresh.predict(sa.inference_view(c))))))
            print(req['fold'],spec['key'],'refit and replay complete',flush=True)
    for k in req['protocol']['kernels']:
        np.testing.assert_array_equal(latent['direct__'+k],latent['difference__'+k])
    assert set(predictions)==set(saved)
    byid={c.id:c for c in val};row_error=0.;loss_error=0.
    expected={(cid,k,g,a) for cid in byid for k in keys for g in PARENTS for a in req['protocol']['gains']}
    assert len(result['rows'])==len(expected)
    assert {(r['cell_id'],r['key'],r['group'],r['gain']) for r in result['rows']}==expected
    for row in result['rows']:
        c=byid[row['cell_id']];p=vp[c.id,row['group']];q=predictions[c.id,row['key']]
        mae=float(np.mean(abs(p+row['gain']*(q-p)-c.y[sa.K:])))
        row_error=max(row_error,abs(mae-row['mae']))
    recomputed=[]
    assert len(result['trials'])==20
    assert {(t['key'],t['gain']) for t in result['trials']}=={(k,a) for k in keys for a in req['protocol']['gains']}
    for trial in result['trials']:
        domain={}
        for c in val:
            p=vp[c.id,'B123'];q=predictions[c.id,trial['key']]
            domain.setdefault(c.domain,[]).append(float(np.mean(abs(p+trial['gain']*(q-p)-c.y[sa.K:]))))
        loss=float(np.mean([np.mean(v) for v in domain.values()]))
        loss_error=max(loss_error,abs(loss-trial['validation_mae']))
        assert trial['target']==trial['key'].split('__')[0]
        recomputed.append(dict(trial,validation_mae=loss))
    for target in req['protocol']['targets']:
        selected=min([r for r in recomputed if r['target']==target],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
        assert selected['key']==result['selected'][target]['key'] and selected['gain']==result['selected'][target]['gain']
    assert max(error,refit_error,row_error,loss_error)<1e-8
    sa.write_json(root/'verification.json',dict(status='PASS',prediction_error=error,refit_error=refit_error,
        row_error=row_error,loss_error=loss_error,refitted_models=4,trials=20,rows=len(expected),
        result_sha256=sa.digest(root/'result.json'),request_sha256=sa.digest(root/'request.json'),
        code_sha256=sa.digest(Path(__file__)),limits='All4GP refits use same training implementation; algebra tested separately. Validation-only audit, not independent held-domain generalization.'))
    print('PASS',req['fold'],flush=True)


if __name__=='__main__':main()
