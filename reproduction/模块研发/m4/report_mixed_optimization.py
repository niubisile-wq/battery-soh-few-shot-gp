"""Report every fitted covariance, including nonconvergence and boundary hits."""
import argparse,json,hashlib
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa,HERE
from evaluate import dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--partial',action='store_true')
    args=ap.parse_args();batchroot=HERE.parent/'results/m4_mixed_effects_batch_v1'
    if args.partial:
        statepath=batchroot/'progress.json';state_bytes=statepath.read_bytes();state=json.loads(state_bytes);folds=state['completed']
    else:
        statepath=batchroot/'result.json';state_bytes=statepath.read_bytes();state=json.loads(state_bytes)
        assert state['status']=='VALIDATION_BATCH_COMPLETE';folds=state['folds'];assert len(folds)==21
    rows=[];errors=[]
    for fold in folds:
        if fold['status']!='VALIDATION_AUDITED':errors.append(fold);continue
        root=Path(fold['branch']);r=json.loads((root/'result.json').read_text());audit=json.loads((root/'verification.json').read_text())
        assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
        for spec in r['models']:
            path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];m=joblib.load(path);gp=m.gp
            assert gp.optimization_==spec['optimization'] and gp.diagnostics_==spec['diagnostics']
            family,kernel=spec['key'].split('__');selected=r['selected'][family]['key']==spec['key']
            rho=float(gp.rho_);trace=float(rho*np.trace(gp.slope_contrast_))
            assert abs(trace-spec['diagnostics']['random_contrast_trace'])<1e-8
            assert abs(rho-spec['diagnostics']['rho'])<1e-12
            theta=gp.kernel_.theta;bounds=gp.kernel_.bounds
            kernelhits=[int(j) for j in range(len(theta)) if min(abs(theta[j]-bounds[j]))<1e-4]
            shared=gp.kernel_.k1(gp.X_train_)
            shared_trace=float(np.trace(gp.H_@shared@gp.H_.T))
            rows.append(dict(fold=fold['fold'],family=family,kernel=kernel,selected=selected,
                selected_gain=r['selected'][family]['gain'] if selected else None,
                success=bool(gp.optimization_['success']),message=gp.optimization_['message'],
                iterations=gp.optimization_['iterations'],rho=rho,
                rho_bound_hit=bool(gp.learn_rho and (rho<=1.0001e-6 or rho>=999.9)),
                kernel_bound_coordinates=json.dumps(kernelhits),random_trace=trace,shared_trace=shared_trace,
                random_to_shared_trace=trace/max(shared_trace,1e-16),
                source_labels=len(m.source_keys),zero_reference_error=r['zero_reference_error']))
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    dump_csv(out/'models.csv',rows)
    summary=dict(status='PARTIAL_OPTIMIZATION_SNAPSHOT' if args.partial else 'ALL_OPTIMIZATION_REPORTED',
        audited_folds=len({r['fold'] for r in rows}),models=len(rows),failures=errors,
        nonconverged=[r for r in rows if not r['success']],rho_bound_hits=[r for r in rows if r['rho_bound_hit']],
        kernel_bound_hits=[r for r in rows if r['kernel_bound_coordinates']!='[]'],
        snapshot=state,source_state_sha256=hashlib.sha256(state_bytes).hexdigest(),code_sha256=sa.digest(Path(__file__)),
        limits='Source covariance diagnostics,not causal decomposition or performance evidence. Trace ratio depends on representation and contrast scale. No exclusion or parameter changes.')
    sa.write_json(out/'summary.json',summary)
    print({k:summary[k] for k in ('status','audited_folds','models')},flush=True)
    print('nonconverged',len(summary['nonconverged']),'rho_bound',len(summary['rho_bound_hits']),'kernel_bound',len(summary['kernel_bound_hits']),flush=True)


if __name__=='__main__':main()
