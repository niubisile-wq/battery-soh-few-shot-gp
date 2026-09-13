"""Replay nested source episodes and complete parent outputs from saved models."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN,rg


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root).resolve()
    manifest=json.loads((root/'audit_manifest.json').read_text());req=json.loads((root/'request.json').read_text());fold=req['fold']
    cells=sa.load_cells(fold.split(':')[0]);tr,va,te=sa.split(cells,fold);byid={c.id:c for c in tr}
    budget={tuple(k) for k in req['source_keys']};assert len(budget)<=1000
    prov=json.loads((HERE.parent/'results/m2_strict_ablation_v1/folds'/fold.replace(':','__')/'provenance.json').read_text())
    old=json.loads((FROZEN/'candidate.json').read_text());entries={r['group']:r for r in old['manifest'] if r['fold']==fold}
    models={};error=0.;count=0
    with threadpool_limits(limits=1):
        for f in manifest['fits']:
            assert sa.digest(root/f['artifact'])==f['sha256'];excluded=tuple(f['excluded'])
            assert {tuple(k) for k in f['fit_keys']}=={k for k in budget if byid[k[0]].domain not in excluded}
            models[excluded,f['group']]=joblib.load(root/f['artifact'])
        episodes={}
        for excluded in sorted({k[0] for k in models}):
            ee=joblib.load(root/('exclude_'+'__'.join(excluded))/'episodes.joblib');episodes[excluded]=ee
            for group in ['Base','Base+M1']:
                for e in ee[group]:
                    c=byid[e['cell_id']];q=e['query_indices'];assert c.domain in excluded
                    assert all((c.id,j) in budget for j in q) and np.array_equal(e['y'],c.y[q])
                    ix=np.r_[np.arange(sa.K),q];hidden=sa.inference_view(replace(c,x=c.x[ix],y=c.y[ix],cycle=c.cycle[ix]))
                    b,r=sa.predict_components(models[excluded,group],hidden,prov['frozen'][group]['mode'])
                    p,_=sa.predict_components(models[excluded,'Physical_control'],hidden,prov['frozen']['Physical_control']['mode'])
                    error=max(error,float(np.max(abs(b-e['base']))),float(abs(r-e['residual'])),float(np.max(abs(p-e['physical']))));count+=1
        heads={};risk_error=0.
        for h in manifest['heads']:
            held=h['held'];group=h['group'];family='Base' if group=='Base+M2' else 'Base+M1'
            risk=[]
            for d in sorted({c.domain for c in tr}):
                if d!=held:risk.extend(e for e in episodes[tuple(sorted([held,d]))][family] if e['domain']==d)
            assert set(h['train_ids'])=={c.id for c in tr if c.domain!=held}
            assert sa.digest(root/h['artifact'])==h['sha256'];head=joblib.load(root/h['artifact']);heads[held,group]=head
            costs=np.array([[rg.expert_risk(e,p) for p in head['params']] for e in risk])
            assert head['source_ids']==[e['cell_id'] for e in risk]
            risk_error=max(risk_error,float(np.max(abs(costs-head['costs']))))
        outputs=joblib.load(root/'complete_parent_source_oof.joblib');assert len(outputs)==len(tr)
        assert sa.digest(root/'complete_parent_source_oof.joblib')==manifest['output_sha256']
        full_error=0.
        for e in outputs:
            c=byid[e['cell_id']];held=c.domain;q=e['query_indices'];ix=np.r_[np.arange(sa.K),q]
            hidden=sa.inference_view(replace(c,x=c.x[ix],y=c.y[ix],cycle=c.cycle[ix]));pp={}
            for group,raw,short in [('Base+M2','B','B2'),('Base+M1+M2','B1','B12')]:
                family='Base' if raw=='B' else 'Base+M1'
                episode=next(v for v in episodes[(held,)][family] if v['cell_id']==c.id)
                sel=entries[group]['selection'];pp[raw]=episode['base']
                pp[short]=rg.attach(episode,hidden,heads[held,group])['reference_predictions'][sel['reference_view']][sel['reference_index']]
            branch=models[(held,),'waveform'].predict(hidden)
            pp.update({g+'3':p+req['gain']*(branch-p) for g,p in list(pp.items())})
            full_error=max(full_error,max(float(np.max(abs(p-e['predictions'][g]))) for g,p in pp.items()))
    assert max(error,risk_error,full_error)<1e-8
    result=dict(status='PASS',fit_models=len(models),replayed_component_episodes=count,source_cells=len(outputs),
        component_error=error,risk_cost_error=risk_error,complete_parent_error=full_error,
        inherited_configuration_limit=req['protocol']['configuration_boundary'],code_sha256=sa.digest(Path(__file__)))
    sa.write_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
