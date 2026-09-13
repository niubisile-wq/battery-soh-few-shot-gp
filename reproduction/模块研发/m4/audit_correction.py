"""Refit selected residuals and independently replay all full-parent validation trials."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from correction import ResidualCorrector
from source_oof import sa


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--correction',required=True);args=ap.parse_args()
    source=Path(args.source);root=Path(args.correction);req=json.loads((source/'request.json').read_text())
    result=json.loads((root/'result.json').read_text());protocol=json.loads((root/'protocol.json').read_text())['protocol']
    cells=sa.load_cells(req['fold'].split(':')[0]);train,val,test=sa.split(cells,req['fold'])
    byid={c.id:sa.source_view(c,[j for cid,j in req['source_keys'] if cid==c.id]) for c in train}
    episodes=joblib.load(source/'complete_parent_source_oof.joblib');vp=joblib.load(root/'validation_parents.joblib')
    assert set(vp)=={(c.id,g) for c in val for g in ['B','B1','B2','B12','B3','B13','B23','B123']}
    assert {e['cell_id'] for e in episodes}==set(byid)
    checks={(g,result['selected']['key']) for g in ['B','B1','B2','B12','B3','B13','B23','B123']}
    checks|={('B123',r['key']) for r in result['trials']};models={};error=0.
    with threadpool_limits(limits=1):
        for g,key in sorted(checks):
            saved=joblib.load(root/'models'/(g+'__'+key+'.joblib'))
            from anchored_correction import AnchoredCorrector
            from condition_correction import ConditionCorrector
            from consensus_correction import ConsensusCorrector
            from robust_correction import RobustCorrector
            from centered_correction import CenteredCorrector
            from within_correction import WithinCorrector
            assert type(saved) in [ResidualCorrector,AnchoredCorrector,ConditionCorrector,ConsensusCorrector,RobustCorrector,CenteredCorrector,WithinCorrector]
            model=type(saved)(saved.view,saved.learner,saved.alpha,saved.clip).fit(episodes,byid,g)
            assert saved.source_keys==model.source_keys
            if isinstance(saved,CenteredCorrector):
                for domain,offset in saved.domain_offsets.items():
                    values=[float(np.mean(e['y']-e['predictions'][g])) for e in episodes if e['domain']==domain]
                    assert abs(offset-sum(values)/len(values))<1e-8
                assert saved.domain_offsets==model.domain_offsets and abs(saved.mean)<1e-10
            assert set(saved.source_keys)<={tuple(k) for k in req['source_keys']}
            if isinstance(saved,ConsensusCorrector) and saved.learner!='constant':
                assert saved.exclusions==model.exclusions
                assert len(saved.members)==len({e['domain'] for e in episodes})
                for member,spec in zip(saved.members,saved.exclusions):
                    expected={(e['cell_id'],int(j)) for e in episodes if e['domain']!=spec['held_domain'] for j in e['query_indices']}
                    assert set(member.source_keys)==expected
                    assert {cid for cid,j in expected}==set(spec['training_ids'])
            for c in val:
                b=vp[c.id,g];error=max(error,float(np.max(abs(model.predict(c,b)-saved.predict(c,b)))))
            models[g,key]=model
        loss_error=0.
        for trial in result['trials']:
            model=models['B123',trial['key']];rr=[]
            for c in val:
                b=vp[c.id,'B123'];pred=b+trial['gain']*model.predict(c,b)
                rr.append(dict(dataset=c.dataset,domain=c.domain,mae=float(np.mean(abs(c.y[sa.K:]-pred)))))
            # Explicit domain-equal recomputation.
            domains={r['domain'] for r in rr};loss=float(np.mean([np.mean([r['mae'] for r in rr if r['domain']==d]) for d in domains]))
            loss_error=max(loss_error,abs(loss-trial['validation_mae']))
        assert result['selected']==min(result['trials'],key=lambda r:(r['validation_mae'],r['gain'],r['key']))
    assert max(error,loss_error)<1e-10
    audit=dict(status='PASS',refitted_models=len(checks),all_full_parent_trials=len(result['trials']),
        prediction_error=error,validation_loss_error=loss_error,result_sha256=sa.digest(root/'result.json'),
        code_sha256=sa.digest(Path(__file__)),limit='Replays selected estimator fits and validation selection, not independent test performance.')
    sa.write_json(root/'verification.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
