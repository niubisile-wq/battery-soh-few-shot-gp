"""Nested source-only complete-parent episodes for fourth-module residual learning."""
import argparse
from collections import defaultdict
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'m3'))
from screen import sa,FROZEN
from waveform import WaveformGP
from conditional_geometry import ConditionalGeometry
import reference_gate as rg


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    fold=args.fold;name=fold.replace(':','__');root=HERE.parent/'results'
    candidate=root/'m3_waveform_candidate_v1';selected=json.loads((candidate/'candidate.json').read_text())
    original=json.loads((FROZEN/'candidate.json').read_text())
    entries={r['group']:r for r in original['manifest'] if r['fold']==fold}
    prov=json.loads((root/'m2_strict_ablation_v1/folds'/name/'provenance.json').read_text())
    sel=json.loads((candidate/'folds'/name/'selection.json').read_text())
    key=sel['original_validation_selection']['key'];wv,wk,wm=key.split('__')
    gain=sel['global_step']*sel['original_validation_selection']['weight']
    cells=sa.load_cells(fold.split(':')[0]);train,val,test=sa.split(cells,fold)
    allowed=defaultdict(list)
    for cid,j in prov['source_keys']:allowed[cid].append(j)
    budget={(cid,j) for cid,ix in allowed.items() for j in ix}
    assert len(budget)<=1000 and set(allowed)=={c.id for c in train}
    domains=sorted({c.domain for c in train});assert len(domains)>=3
    assert set(allowed).isdisjoint({c.id for c in val+test})
    protocol=json.loads((HERE/'protocol.json').read_text())
    sa.write_json(out/'request.json',dict(status='RUNNING',fold=fold,protocol=protocol,
        code_sha256=sa.digest(Path(__file__)),m3_manifest_sha256=sa.digest(candidate/'candidate.json'),
        branch_key=key,gain=gain,source_keys=sorted(budget),no_outer_query_scoring=True))
    fits=[];episodes={};branches={}
    with threadpool_limits(limits=1):
        for excluded in [(d,) for d in domains]+list(combinations(domains,2)):
            fit=[c for c in train if c.domain not in excluded];idx={c.id:allowed[c.id] for c in fit}
            visible=[sa.source_view(c,idx[c.id]) for c in fit];models={}
            folder=out/('exclude_'+'__'.join(excluded));folder.mkdir()
            for group in sa.JOBS:
                f=prov['frozen'][group]
                model=sa.GPModel(sa.SPECS[f['candidate']],f['config']).fit(visible,idx)
                assert len(model.gp.X_train_)==sum(len(v) for v in idx.values())
                path=folder/(group+'.joblib');joblib.dump(model,path,compress=3);models[group]=model
                fits.append(dict(excluded=list(excluded),group=group,fit_keys=sorted((cid,j) for cid,ii in idx.items() for j in ii),
                    artifact=str(path.relative_to(out)),sha256=sa.digest(path)))
            if len(excluded)==1:
                gp=WaveformGP(wv,wk).fit(visible,idx);branch=ConditionalGeometry(gp,wm)
                path=folder/'waveform.joblib';joblib.dump(branch,path,compress=3)
                fits.append(dict(excluded=list(excluded),group='waveform',fit_keys=sorted(gp.source_keys),artifact=str(path.relative_to(out)),sha256=sa.digest(path)))
            ee={'Base':[],'Base+M1':[]}
            for c in train:
                if c.domain not in excluded:continue
                query=np.array([j for j in allowed[c.id] if j>=sa.K]);assert len(query)
                ix=np.r_[np.arange(sa.K),query]
                sparse=replace(c,x=c.x[ix],y=c.y[ix],cycle=c.cycle[ix]);hidden=sa.inference_view(sparse)
                comp={g:sa.predict_components(m,hidden,prov['frozen'][g]['mode']) for g,m in models.items()}
                for g in ee:
                    b,r=comp[g]
                    ee[g].append(dict(cell_id=c.id,dataset=c.dataset,domain=c.domain,query_indices=query.tolist(),
                        y=c.y[query],base=b,residual=r,physical=comp['Physical_control'][0]))
                if len(excluded)==1:branches[c.id]=branch.predict(hidden)
            joblib.dump(ee,folder/'episodes.joblib',compress=3);episodes[excluded]=ee
            sa.write_json(out/'fit_progress.json',dict(fits=fits))
            print(fold,'excluded',excluded,'GPs and episodes saved',flush=True)
        outputs=[];head_manifest=[]
        for held in domains:
            risk_cells=[sa.source_view(c,allowed[c.id]) for c in train if c.domain!=held]
            pred={}
            for family,group,short in [('Base','Base+M2','B2'),('Base+M1','Base+M1+M2','B12')]:
                risk=[]
                for d in domains:
                    if d!=held:risk.extend(e for e in episodes[tuple(sorted([held,d]))][family] if e['domain']==d)
                assert all(e['domain']!=held for e in risk)
                head=rg.fit(risk,risk_cells,metric=True,bagged=True)
                path=out/('head__'+held+'__'+group+'.joblib');joblib.dump(head,path,compress=3)
                head_manifest.append(dict(held=held,group=group,train_ids=[e['cell_id'] for e in risk],artifact=str(path.relative_to(out)),sha256=sa.digest(path)))
                selection=entries[group]['selection'];view=selection['reference_view'];index=selection['reference_index']
                for e in episodes[(held,)][family]:
                    c=next(c for c in train if c.id==e['cell_id'])
                    pp=rg.attach(e,sa.inference_view(c),head)['reference_predictions'][view][index]
                    pred[c.id,short]=pp
            a={e['cell_id']:e for e in episodes[(held,)]['Base']};b={e['cell_id']:e for e in episodes[(held,)]['Base+M1']}
            for cid,e in a.items():
                pp=dict(B=e['base'],B1=b[cid]['base'],B2=pred[cid,'B2'],B12=pred[cid,'B12'])
                pp.update({g+'3':p+gain*(branches[cid]-p) for g,p in list(pp.items())})
                assert all(np.isfinite(p).all() and p.shape==e['y'].shape for p in pp.values())
                outputs.append(dict(cell_id=cid,dataset=e['dataset'],domain=held,query_indices=e['query_indices'],y=e['y'],predictions=pp))
        assert len(outputs)==len(train)
        joblib.dump(outputs,out/'complete_parent_source_oof.joblib',compress=3)
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in original['manifest'])
    assert all(sa.digest(candidate/r['artifact'])==r['sha256'] for r in selected['manifest'])
    sa.write_json(out/'audit_manifest.json',dict(status='SOURCE_OOF_COMPLETE_AUDIT_PENDING',fold=fold,fits=fits,heads=head_manifest,
        source_cells=len(outputs),source_query_rows=sum(len(e['y']) for e in outputs),output_sha256=sa.digest(out/'complete_parent_source_oof.joblib'),
        limits=protocol['configuration_boundary']))
    print(fold,'complete-parent nested source OOF generated; independent audit pending',flush=True)


if __name__=='__main__':main()
