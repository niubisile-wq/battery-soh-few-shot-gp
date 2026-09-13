"""All-four-model source refit and independent validation calculation."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from mixed_effects_pilot import load_inputs
from mixed_effects_branch import MixedEffectsBranch
from score import PARENTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);root=Path(ap.parse_args().root)
    req=json.loads((root/'request.json').read_text());r=json.loads((root/'result.json').read_text())
    assert r['status']=='VALIDATION_COMPLETE_AUDIT_PENDING'
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    source,val,allowed,vp,refs,origin=load_inputs(req['fold']);assert origin==req['origin']
    predpath=root/'validation_branch_predictions.joblib';assert sa.digest(predpath)==r['prediction_sha256']
    saved=joblib.load(predpath);pred={};error=0.;refit_error=0.
    keys={f+'__'+k for f in req['protocol']['families'] for k in req['protocol']['kernels']}
    assert len(r['models'])==4 and {s['key'] for s in r['models']}==keys
    with threadpool_limits(limits=1):
        for spec in r['models']:
            path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];m=joblib.load(path)
            family,kernel=spec['key'].split('__');assert m.gp.learn_rho==(family=='mixed')
            assert spec['optimization']==m.gp.optimization_ and spec['diagnostics']==m.gp.diagnostics_
            fresh=MixedEffectsBranch.from_encoder(refs[kernel],source,allowed,learn_rho=family=='mixed')
            assert fresh.source_keys==m.source_keys
            np.testing.assert_allclose(fresh.gp.X_train_,m.gp.X_train_,atol=1e-12,rtol=0)
            for c in val:
                p=m.predict(sa.inference_view(c));pred[c.id,spec['key']]=p
                error=max(error,float(np.max(abs(p-saved[c.id,spec['key']]))))
                refit_error=max(refit_error,float(np.max(abs(p-fresh.predict(sa.inference_view(c))))))
            print(req['fold'],spec['key'],'refitted',flush=True)
    assert set(pred)==set(saved)
    byid={c.id:c for c in val};losses={};row_error=0.;trial_error=0.
    expected={(cid,k,g,a) for cid in byid for k in keys for g in PARENTS for a in req['protocol']['gains']}
    assert len(r['rows'])==len(expected) and {(v['cell_id'],v['key'],v['group'],v['gain']) for v in r['rows']}==expected
    for row in r['rows']:
        cid,key,g,gain=row['cell_id'],row['key'],row['group'],row['gain'];c=byid[cid]
        mae=float(np.mean(abs((1-gain)*vp[cid,g]+gain*pred[cid,key]-c.y[sa.K:])))
        row_error=max(row_error,abs(mae-row['mae']))
        if g=='B123':losses.setdefault((key,gain),{}).setdefault(c.domain,[]).append(mae)
    assert len(r['trials'])==20 and {(v['key'],v['gain']) for v in r['trials']}==set(losses)
    trials=[]
    for trial in r['trials']:
        loss=float(np.mean([np.mean(v) for v in losses[trial['key'],trial['gain']].values()]))
        trial_error=max(trial_error,abs(loss-trial['validation_mae']))
        assert trial['target']==trial['key'].split('__')[0]
        trials.append(dict(trial,validation_mae=loss))
    for family in req['protocol']['families']:
        best=min([t for t in trials if t['target']==family],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
        assert best['key']==r['selected'][family]['key'] and best['gain']==r['selected'][family]['gain']
    assert max(error,refit_error,row_error,trial_error)<1e-8
    sa.write_json(root/'verification.json',dict(status='PASS',prediction_error=error,refit_error=refit_error,
        row_error=row_error,trial_error=trial_error,models_refitted=4,trials=20,rows=len(expected),
        request_sha256=sa.digest(root/'request.json'),result_sha256=sa.digest(root/'result.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Same training implementation refits; original source encoder separately audited,not refitted here. No held-domain performance confirmation.'))
    print('PASS',req['fold'],flush=True)


if __name__=='__main__':main()
