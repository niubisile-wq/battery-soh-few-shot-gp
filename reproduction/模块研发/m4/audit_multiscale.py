"""Refit selected direct branch, replay all validation branches and selection."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from multiscale import MultiscaleGP
from conditional_geometry import ConditionalGeometry


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);root=Path(ap.parse_args().root)
    req=json.loads((root/'request.json').read_text());result=json.loads((root/'result.json').read_text())
    estimator=MultiscaleGP
    if req.get('structured',False):
        from structured_multiscale import StructuredMultiscaleGP
        estimator=StructuredMultiscaleGP
    if req.get('registered',False):
        from registered_charge import RegisteredChargeGP
        estimator=RegisteredChargeGP
    assert result['status']=='VALIDATION_COMPLETE_AUDIT_PENDING'
    parentpath=Path(req['validation_parent_path']);assert sa.digest(parentpath)==req['validation_parent_sha256']
    predpath=root/'validation_branch_predictions.joblib';assert sa.digest(predpath)==result['prediction_sha256']
    vp=joblib.load(parentpath);saved=joblib.load(predpath)
    cells=sa.load_cells(req['fold'].split(':')[0]);train,val,test=sa.split(cells,req['fold'])
    allowed={}
    for cid,j in req['source_keys']:allowed.setdefault(cid,[]).append(j)
    assert set(allowed)=={c.id for c in train}
    visible=[sa.source_view(c,allowed[c.id]) for c in train]
    selected_encoder,selected_kernel,_=result['selected']['key'].split('__')
    predictions={};error=0.;refit_error=0.
    with threadpool_limits(limits=1):
        for spec in result['models']:
            path=Path(spec['path']);assert sa.digest(path)==spec['sha256']
            gp=joblib.load(path)
            assert set(gp.source_keys)=={tuple(k) for k in req['source_keys']}
            assert len(gp.gp.X_train_)==len(gp.source_keys)<=1000
            refit=None
            if (spec['encoder'],spec['kernel'])==(selected_encoder,selected_kernel):
                refit=estimator(selected_encoder,selected_kernel).fit(visible,allowed)
                assert refit.source_keys==gp.source_keys
            for mode in req['protocol']['modes']:
                branch=ConditionalGeometry(gp,mode);key='__'.join((spec['encoder'],spec['kernel'],mode))
                for c in val:
                    p=branch.predict(sa.inference_view(c));predictions[c.id,key]=p
                    error=max(error,float(np.max(abs(p-saved[c.id,key]))))
                    if refit is not None:
                        refit_error=max(refit_error,float(np.max(abs(p-ConditionalGeometry(refit,mode).predict(sa.inference_view(c))))))
        loss_error=0.
        for trial in result['trials']:
            domains={c.domain for c in val};values={d:[] for d in domains}
            for c in val:
                p=vp[c.id,'B123'];q=predictions[c.id,trial['key']]
                values[c.domain].append(float(np.mean(abs(p+trial['gain']*(q-p)-c.y[sa.K:]))))
            loss=float(np.mean([np.mean(v) for v in values.values()]))
            loss_error=max(loss_error,abs(loss-trial['validation_mae']))
    assert set(predictions)==set(saved)
    assert result['selected']==min(result['trials'],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
    assert max(error,refit_error,loss_error)<1e-8
    sa.write_json(root/'verification.json',dict(status='PASS',prediction_error=error,selected_refit_error=refit_error,
        validation_loss_error=loss_error,branches_replayed=8,trials_recomputed=40,refitted_gp_models=1,
        result_sha256=sa.digest(root/'result.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Refits selected GP only; replays every validation branch and40full-parent selection trials. Does not independently refit all4GPs or score outer queries.'))
    print('PASS selected GP refit and all validation branches',flush=True)


if __name__=='__main__':main()
