"""Fit first correction grid and score validation only; no outer query access."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN
from correction import ResidualCorrector


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--variant',choices=['plain','anchored','condition','consensus','robust','centered','within'],default='plain');args=ap.parse_args()
    src=Path(args.source).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assert json.loads((src/'verification.json').read_text())['status']=='PASS'
    req=json.loads((src/'request.json').read_text());fold=req['fold']
    if args.variant=='anchored':
        from anchored_correction import AnchoredCorrector
        estimator=AnchoredCorrector;protocol_path=HERE/'anchored_protocol.json'
    elif args.variant=='condition':
        from condition_correction import ConditionCorrector
        estimator=ConditionCorrector;protocol_path=HERE/'condition_protocol.json'
    elif args.variant=='consensus':
        from consensus_correction import ConsensusCorrector
        estimator=ConsensusCorrector;protocol_path=HERE/'consensus_protocol.json'
    elif args.variant=='robust':
        from robust_correction import RobustCorrector
        estimator=RobustCorrector;protocol_path=HERE/'robust_protocol.json'
    elif args.variant=='centered':
        from centered_correction import CenteredCorrector
        estimator=CenteredCorrector;protocol_path=HERE/'centered_protocol.json'
    elif args.variant=='within':
        from within_correction import WithinCorrector
        estimator=WithinCorrector;protocol_path=HERE/'within_protocol.json'
    else:estimator=ResidualCorrector;protocol_path=HERE/'correction_protocol.json'
    p=json.loads(protocol_path.read_text());sa.write_json(out/'protocol.json',dict(protocol=p,variant=args.variant,fold=fold,source=str(src),code_sha256=sa.digest(Path(__file__))))
    episodes=joblib.load(src/'complete_parent_source_oof.joblib');cells=sa.load_cells(fold.split(':')[0]);tr,val,test=sa.split(cells,fold)
    byid={c.id:sa.source_view(c,[j for cid,j in req['source_keys'] if cid==c.id]) for c in tr}
    candidate=HERE.parent/'results/m3_waveform_candidate_v1';obj=json.loads((candidate/'candidate.json').read_text())
    adapters={r['group']:joblib.load(candidate/r['artifact']) for r in obj['manifest'] if r['fold']==fold}
    groups=['B','B1','B2','B12','B3','B13','B23','B123'];vp={};rows=[];models={};boundary=0.
    specs=[('constant','state','constant',1)]+[(view+'__'+learner+'__'+str(alpha),view,learner,alpha) for view in p.get('features',p.get('views')) for learner,aa in p['learners'].items() for alpha in aa]
    with threadpool_limits(limits=1):
        for c in val:
            for g in groups:
                if g.endswith('3'):v=adapters[g].predict(sa.inference_view(c))
                else:
                    a=adapters[g+'3'];v=a.parent.predict(sa.inference_view(c)) if a.parent_mode is None else a.parent.predict(sa.inference_view(c),a.parent_mode)
                vp[c.id,g]=v
        joblib.dump(vp,out/'validation_parents.joblib')
        for key,view,learner,alpha in specs:
            for g in groups:
                model=estimator(view,learner,alpha,p['clip_soh']).fit(episodes,byid,g)
                path=out/'models'/(g+'__'+key+'.joblib');path.parent.mkdir(exist_ok=True);joblib.dump(model,path);models[g,key]=model
                for c in val:
                    base=vp[c.id,g];r=model.predict(c,base)
                    yy=c.y.copy();yy[sa.K:]=123
                    boundary=max(boundary,float(np.max(abs(r-model.predict(replace(c,y=yy),base)))))
                    n=min(len(c.x),sa.K+3)
                    boundary=max(boundary,float(np.max(abs(r[:n-sa.K]-model.predict(sa.prefix(c,n),base[:n-sa.K])))))
                    for gain in p['gains']:
                        rows.append(dict(key=key,group=g,gain=gain,cell_id=c.id,dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(base+gain*r-c.y[sa.K:])))))
            print(fold,key,'all8 residuals fitted and validation scored',flush=True)
    assert boundary<1e-8
    trials=[]
    for key,*_ in specs:
        for gain in p['gains']:
            rr=[r for r in rows if r['key']==key and r['gain']==gain and r['group']=='B123']
            trials.append(dict(key=key,gain=gain,validation_mae=sa.macro(rr)))
    selected=min(trials,key=lambda r:(r['validation_mae'],r['gain'],r['key']))
    identity=min(r['validation_mae'] for r in trials if r['gain']==0)
    sa.write_json(out/'result.json',dict(status='VALIDATION_PILOT_COMPLETE_NOT_ADOPTED',selected=selected,identity_validation_mae=identity,
        trials=trials,rows=rows,max_boundary_error=boundary,models=len(models),limits=p['limits']))
    print(json.dumps(dict(selected=selected,identity_validation_mae=identity,boundary=boundary)),flush=True)


if __name__=='__main__':main()
