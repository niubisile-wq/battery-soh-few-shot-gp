"""Full84-model source optimizer report, never excludes boundary/nonconverged fits."""
import json
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa,HERE
from evaluate import dump_csv


def main():
    results=HERE.parent/'results';batchpath=results/'m4_hierarchical_batch_v1/result.json'
    batch=json.loads(batchpath.read_text());assert batch['status']=='VALIDATION_BATCH_COMPLETE' and len(batch['folds'])==21
    rows=[]
    for fold in batch['folds']:
        root=Path(fold['branch']);r=json.loads((root/'result.json').read_text());audit=json.loads((root/'verification.json').read_text())
        assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
        for spec in r['models']:
            path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];model=joblib.load(path);gp=model.parent.gp
            assert spec['optimization']==gp.optimization_ and spec['diagnostics']==gp.diagnostics_
            family,kernel=spec['key'].split('__')
            if family=='hierarchical':
                rhos=[gp.rho_cell_,gp.rho_domain_]
                traces=[float(v*np.trace(cov)) for v,cov in zip(rhos,gp.nuisance_contrasts_)]
                assert abs(gp.rho_-sum(rhos))<1e-12
            else:rhos=[gp.rho_,0.];traces=[float(gp.rho_*np.trace(gp.slope_contrast_)),0.]
            theta,bounds=gp.kernel_.theta,gp.kernel_.bounds
            hits=[int(i) for i in range(len(theta)) if min(abs(theta[i]-bounds[i]))<1e-4]
            rows.append(dict(fold=fold['fold'],family=family,kernel=kernel,success=bool(gp.optimization_['success']),
                iterations=gp.optimization_['iterations'],message=gp.optimization_['message'],
                selected=r['selected'][family]['key']==spec['key'],selected_gain=r['selected'][family]['gain'],
                rho_cell=rhos[0],rho_domain=rhos[1],target_rho=gp.rho_,
                rho_bound_hit=any(v<=1.0001e-6 or v>=999.9 for v in (rhos if family=='hierarchical' else rhos[:1])),
                cell_contrast_trace=traces[0],domain_contrast_trace=traces[1],kernel_bound_coordinates=json.dumps(hits),
                source_labels=len(model.source_keys),control_replay_error=r['control_error']))
    assert len(rows)==84
    out=results/'m4_hierarchical_optimization_v1';out.mkdir(exist_ok=False)
    dump_csv(out/'models.csv',rows)
    summary=dict(status='ALL84_SOURCE_MODELS_REPORTED',models=84,
        nonconverged=[r for r in rows if not r['success']],rho_bound_hits=[r for r in rows if r['rho_bound_hit']],
        kernel_bound_hits=[r for r in rows if r['kernel_bound_coordinates']!='[]'],
        source_sha256=sa.digest(batchpath),code_sha256=sa.digest(Path(__file__)),
        limits='Source numerical diagnostics only,not causal variance fractions or predictive acceptance. No exclusions or re-tuning.')
    sa.write_json(out/'summary.json',summary)
    print('84models', {k:len(summary[k]) for k in ('nonconverged','rho_bound_hits','kernel_bound_hits')},flush=True)


if __name__=='__main__':main()
