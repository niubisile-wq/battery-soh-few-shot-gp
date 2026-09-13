"""Freeze and independently replay the central passing development step (0.5)."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa,HERE,FROZEN,GROUPS,error_metrics,aggregate,dump_csv
from waveform_adapter import WaveformM3Adapter
from summarize_strict import paired_stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    family='wave_pair_pca8';step=.5;source=HERE.parent/'results/m3_waveform_screen_v1';controls=source/'matched_controls'
    sensitivity=json.loads((HERE.parent/'results/m3_waveform_shrink_v1/summary.json').read_text())
    assert family+'__'+str(step) in sensitivity['passing_point_candidates']
    old=json.loads((FROZEN/'candidate.json').read_text());assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in old['manifest'])
    sa.write_json(out/'request.json',dict(family=family,global_step=step,status='DEVELOPMENT_CANDIDATE_AUDIT',
        selection_disclosure='Global step selected after development sensitivity; per-fold branch/weight selected using validation only. Not independent test.',
        remaining=['Stability checks','Mechanism/control review','Final acceptance audit']))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)]
    rows=[];manifest=[];max_cache_error=0.;max_replay_error=0.;boundary=0.
    with threadpool_limits(limits=1):
        for fold in sorted({r['fold'] for r in old['manifest']}):
            train,val,test=sa.split(cells,fold);name=fold.replace(':','__');dest=out/'folds'/name
            selected=json.loads((controls/'folds'/name/'selection.json').read_text())[family]
            branchpath=source/'folds'/name/(selected['key']+'.joblib');branch=joblib.load(branchpath)
            entries={r['group']:r for r in old['manifest'] if r['fold']==fold}
            a0=joblib.load(FROZEN/entries['Base+M2']['artifact']);a1=joblib.load(FROZEN/entries['Base+M1+M2']['artifact'])
            budget={tuple(k) for k in entries['Base+M2']['source_keys']};assert set(branch.source_keys)==budget and len(budget)<=1000
            parents={'B':(a0.parent,a0.parent_mode),'B1':(a1.parent,a1.parent_mode),'B2':(a0,None),'B12':(a1,None)}
            adapters={g+'3':WaveformM3Adapter(parent,branch,step*selected['weight'],mode) for g,(parent,mode) in parents.items()}
            for g,adapter in adapters.items():
                path=dest/g/'adapter.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(adapter,path)
                manifest.append(dict(fold=fold,group=g,artifact=str(path.relative_to(out)),sha256=sa.digest(path),
                    branch_source=str(branchpath),branch_sha256=sa.digest(branchpath),source_keys=sorted(budget),
                    validation_weight=selected['weight'],global_step=step,effective_weight=adapter.gain))
            sa.write_json(dest/'selection.json',dict(original_validation_selection=selected,global_step=step,
                validation_cells=[c.id for c in val],source_keys=sorted(budget)))
            loaded={g:joblib.load(dest/g/'adapter.joblib') for g in adapters}
            for c in test:
                pp={}
                for g,(parent,mode) in parents.items():pp[g]=parent.predict(sa.inference_view(c)) if mode is None else parent.predict(sa.inference_view(c),mode)
                for g,adapter in adapters.items():
                    p=adapter.predict(c);pp[g]=p
                    max_replay_error=max(max_replay_error,float(np.max(abs(p-loaded[g].predict(c)))))
                    yy=c.y.copy();yy[sa.K:]=123
                    boundary=max(boundary,float(np.max(abs(p-adapter.predict(replace(c,y=yy))))))
                    for n in sorted({sa.K+1,min(len(c.x),sa.K+17)}):
                        boundary=max(boundary,float(np.max(abs(p[:n-sa.K]-adapter.predict(sa.prefix(c,n))))))
                with np.load(controls/'folds'/name/'predictions'/(c.id+'.npz')) as z:
                    assert np.array_equal(z['y'],c.y[sa.K:])
                    for g in GROUPS:
                        expected=z[g]+step*(z[g+'3__'+family]-z[g])
                        max_cache_error=max(max_cache_error,float(np.max(abs(pp[g+'3']-expected))),float(np.max(abs(pp[g]-z[g]))))
                assert max(max_cache_error,max_replay_error,boundary)<1e-8
                path=dest/'predictions'/(c.id+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=c.y[sa.K:],**pp)
                for g,p in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(c.y[sa.K:],p)))
            print(fold,'candidate exported, full models replayed, boundaries verified',flush=True)
    assert len(rows)==2920 and len(manifest)==84
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    edges=[('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]
    gates={a+'<'+b:all(look[ds,a][m]<look[ds,b][m] for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae']) for a,b in edges}
    assert all(gates.values());pairs=[]
    for a,b in edges:
        for ds in ['XJTU','MATR','Tongji']:
            aa={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==a};bb={r['cell_id']:r for r in rows if r['dataset']==ds and r['group']==b}
            for m in ['mae','rmse','p95_ae']:pairs.append(dict(comparison=a+'-'+b,dataset=ds,metric=m,**paired_stats(
                [(aa[c][m]-bb[c][m])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in old['manifest'])
    snapshots={}
    for folder in ['m1','m2','m3']:
        for path in (HERE.parent/folder).glob('*.py'):
            target=out/'source_snapshot'/folder/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            snapshots[str(target.relative_to(out))]=sa.digest(target)
    data_hashes={c.id:dict(path=c.path,sha256=sa.digest(sa.ROOT/c.path)) for c in cells}
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'candidate.json',dict(status='POINT_VERIFIED_STABILITY_PENDING',family=family,global_step=step,table=table,gates=gates,
        paired_comparisons=pairs,manifest=manifest,source_snapshot=snapshots,data_hashes=data_hashes,
        max_cache_error=max_cache_error,max_serialized_replay_error=max_replay_error,max_boundary_error=boundary,
        limits=['Exploratory global-step selection on development data, not independent confirmation.',
                'PCA itself is not novel; comparative/absolute reference representation needs mechanism review.',
                'Point improvements do not imply all-cell improvement or statistical significance.']))
    print('POINT VERIFIED: all7 gates,21 folds,365 cells,84 adapters; stability pending',flush=True)


if __name__=='__main__':main()
