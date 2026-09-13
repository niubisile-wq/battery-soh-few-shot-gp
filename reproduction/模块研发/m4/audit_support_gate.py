"""Recalculate source scales and all validation gates using scipy distances."""
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registered',action='store_true');args=ap.parse_args()
    root=HERE.parent/'results'/('m4_registered_gate_validation_v1' if args.registered else 'm4_support_gate_validation_v1')
    result=json.loads((root/'result.json').read_text());error=scale_error=gate_error=0.
    assert len(result['selections'])==21
    with threadpool_limits(limits=1):
        for s in result['selections']:
            path=Path(s['result']);assert sa.digest(path)==s['result_sha256']
            r=json.loads(path.read_text());source=Path(r['source_branch'])
            req=json.loads((source/'request.json').read_text())
            vp=joblib.load(req['validation_parent_path']);bp=joblib.load(source/'validation_branch_predictions.joblib')
            assert sa.digest(Path(req['validation_parent_path']))==req['validation_parent_sha256']
            original=json.loads((source/'result.json').read_text());assert sa.digest(source/'validation_branch_predictions.joblib')==original['prediction_sha256']
            _,val,_=sa.split(sa.load_cells(s['fold'].split(':')[0]),s['fold']);gg={}
            for stem,spec in r['gates'].items():
                assert sa.digest(Path(spec['path']))==spec['sha256']
                gate=joblib.load(spec['path']);model=gate.model
                assert set(model.source_keys)=={tuple(k) for k in req['source_keys']}
                z=model.gp.X_train_;ids=np.array([cid for cid,j in model.source_keys]);dist=cdist(z,z,'sqeuclidean')
                dist[ids[:,None]==ids[None,:]]=np.inf
                nearest=dist.min(1);scales=[float(np.median(nearest[ids==cid])) for cid in sorted(set(ids))]
                scale=max(float(np.median(scales)),1e-8)
                scale_error=max(scale_error,abs(scale-gate.scale2));assert np.isclose(scale,gate.scale2,rtol=1e-7,atol=1e-10)
                for c in val:
                    latent=model.scaler.transform(model.feature_matrix(sa.inference_view(c)))[sa.K:]
                    nearest=cdist(latent,z,'sqeuclidean').min(1)
                    for multiplier in (.25,1,4):
                        weight=(scale*multiplier)/(scale*multiplier+nearest)
                        gate_error=max(gate_error,float(np.max(abs(weight-gate.predict(c,multiplier)))))
                        gg[c.id,stem,multiplier]=weight
            for t in r['trials']:
                stem='__'.join(t['key'].split('__')[:2]);values={d:[] for d in {c.domain for c in val}}
                for c in val:
                    p=vp[c.id,'B123'];q=bp[c.id,t['key']]
                    pred=p+t['gain']*gg[c.id,stem,t['multiplier']]*(q-p)
                    values[c.domain].append(float(np.mean(abs(pred-c.y[sa.K:]))))
                loss=float(np.mean([np.mean(v) for v in values.values()]));error=max(error,abs(loss-t['validation_mae']))
            assert s['selected']==r['selected']==min(r['trials'],key=lambda t:(t['validation_mae'],t['gain'],t['key'],t['multiplier']))
            assert len(r['trials'])==120
            print(s['fold'],'all120 validation objectives and source scales replayed',flush=True)
    assert max(error,gate_error)<1e-8
    sa.write_json(root/'verification.json',dict(status='PASS',folds=21,trials=2520,scale_error=scale_error,
        gate_error=gate_error,validation_loss_error=error,result_sha256=sa.digest(root/'result.json'),
        code_sha256=sa.digest(Path(__file__)),limits='Recalculates source distances and all validation objectives,does not refit GP encoders or establish generalization.'))


if __name__=='__main__':main()
