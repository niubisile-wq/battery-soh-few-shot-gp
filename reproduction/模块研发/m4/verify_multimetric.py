"""Audit multi-metric selection arithmetic, selected validation and outer replay."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from score import PARENTS,addition_edges
from summarize_strict import paired_stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root)
    summary=json.loads((root/'summary.json').read_text());selections=json.loads((root/'frozen_selections.json').read_text())['selections']
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];candidate=HERE.parent/'results/m3_waveform_candidate_v1'
    saved={(r['cell_id'],r['group']):r for r in csv.DictReader((root/'cells.csv').open())};assert len(saved)==8760
    rows=[];replay=0.;metricerr=0.;valerr=0.;keys=['mae','rmse','p95_ae']
    with threadpool_limits(limits=1):
        for s in selections:
            fold=s['fold'];name=fold.replace(':','__');_,val,test=sa.split(cells,fold)
            assert set(s['validation_ids'])=={c.id for c in val}
            for trial in s['trials']:
                score=np.mean([trial['metrics'][g][k]/max(s['baseline'][g][k],1e-10) for g in ['B','B123'] for k in keys])
                assert abs(score-trial['score'])<1e-12
                feasible=all(trial['metrics'][g][k]<=s['baseline'][g][k]+1e-12 for g in ['B','B123'] for k in keys)
                assert feasible==trial['feasible']
            models={}
            for policy,sel in s['selected'].items():
                candidates=s['trials'] if policy=='balanced' else [r for r in s['trials'] if r['feasible']]
                assert sel['choice']==min(candidates,key=lambda r:(r['score'],r['gain'],r['family'],r['key']))
                models[policy]={}
                for g,spec in sel['models'].items():
                    path=Path(spec['path']);assert sa.digest(path)==spec['sha256'];models[policy][g]=joblib.load(path)
                folder=Path(sel['models']['B']['path']).parent.parent;vp=joblib.load(folder/'validation_parents.joblib')
                for g in ['B','B123']:
                    rr=[]
                    for c in val:
                        b=vp[c.id,g];p=b+sel['choice']['gain']*models[policy][g].predict(sa.inference_view(c),b)
                        rr.append(dict(dataset=c.dataset,domain=c.domain,**sa.metrics(c.y[sa.K:],p)))
                    for k in keys:valerr=max(valerr,abs(sa.macro(rr,k)-sel['choice']['metrics'][g][k]))
            for c in test:
                tail=Path('folds')/name/'predictions'/(c.id+'.npz')
                with np.load(root/tail) as z,np.load(candidate/tail) as old:
                    assert np.array_equal(z['y'],c.y[sa.K:]);expected={g:old[g] for g in PARENTS}
                    for policy,mm in models.items():
                        gain=s['selected'][policy]['choice']['gain']
                        for g,m in mm.items():expected[g+'4__'+policy]=old[g]+gain*m.predict(sa.inference_view(c),old[g])
                    for g,p in expected.items():
                        replay=max(replay,float(np.max(abs(p-z[g]))));e=(np.asarray(z[g],float)-np.asarray(z['y'],float))*100
                        metrics=dict(mae=float(abs(e).mean()),rmse=float(np.sqrt(np.dot(e,e)/len(e))),p95_ae=float(np.percentile(abs(e),95)))
                        metricerr=max(metricerr,max(abs(metrics[k]-float(saved[c.id,g][k])) for k in keys))
                        rows.append(dict(cell_id=c.id,domain=c.domain,dataset=c.dataset,group=g,**metrics))
            print(fold,'selected validation and all policy outputs replayed',flush=True)
    assert len(rows)==8760 and max(valerr,replay,metricerr)<1e-8
    table={}
    for r in summary['table']:
        rr=[v for v in rows if v['group']==r['group'] and v['dataset']==r['dataset']];domains={v['domain'] for v in rr}
        mm={k:float(np.mean([np.mean([v[k] for v in rr if v['domain']==d]) for d in domains])) for k in keys}
        assert all(abs(mm[k]-r[k])<1e-8 for k in keys);table[r['dataset'],r['group']]=mm
    pairs=[]
    for policy in ['balanced','constrained']:
        tag=lambda g:g+'__'+policy if '4' in g else g
        for a,b in addition_edges():
            assert summary['gates'][policy][a+'<'+b]==all(table[ds,tag(a)][k]<table[ds,tag(b)][k] for ds in ['XJTU','MATR','Tongji'] for k in keys)
        for a,b in [('B4','B'),('B1234','B123'),('B1234','B124'),('B1234','B134'),('B1234','B234')]:
            for ds in ['XJTU','MATR','Tongji']:
                aa={v['cell_id']:v for v in rows if v['dataset']==ds and v['group']==tag(a)};bb={v['cell_id']:v for v in rows if v['dataset']==ds and v['group']==tag(b)}
                for k in keys:pairs.append(dict(policy=policy,comparison=a+'-'+b,dataset=ds,metric=k,**paired_stats([(aa[c][k]-bb[c][k])/100 for c in aa],[aa[c]['domain'] for c in aa])))
    sa.write_json(root/'verification.json',dict(status='PASS',rows=len(rows),prediction_error=replay,metric_error=metricerr,selected_validation_error=valerr,
        paired_comparisons=pairs,summary_sha256=sa.digest(root/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Unselected trial prediction arrays not independently replayed; all trial objective/constraint arithmetic and selected validation predictions verified. Development intervals without selection correction.'))


if __name__=='__main__':main()
