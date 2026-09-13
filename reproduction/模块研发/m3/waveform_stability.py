"""Fixed candidate M3, paired original-v3 risk resampling; no candidate tuning."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from screen import sa, HERE, FROZEN, SCREEN, aggregate, error_metrics, dump_csv
from reference_sensitivity import reweight, risk_seeds
import reference_gate as rg

EDGES = [('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),
         ('B123','B3'),('B13','B1'),('B23','B2')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    candidate=HERE.parent/'results/m3_waveform_candidate_v1'
    obj=json.loads((candidate/'candidate.json').read_text())
    old=json.loads((FROZEN/'candidate.json').read_text())
    seeds=risk_seeds()
    sa.write_json(out/'protocol.json',dict(status='RUNNING',seeds=seeds,
        candidate_sha256=sa.digest(candidate/'candidate.json'),code_sha256=sa.digest(Path(__file__)),
        fixed='M3 branch, validation weight, global step 0.5, all GP fits and M2 settings.',
        change='Original v3 within-source-domain risk resampling, paired with the same draw in controls.',
        limits='Development sensitivity, not independent tests or ten full model retrainings.'))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)]
    byid={c.id:c for c in cells};rows=[];resamples=[];replay=0.
    previous=json.loads((HERE.parent/'results/m2_reference_candidate_v3_risk_sensitivity/summary.json').read_text())
    prior={(r['seed'],r['cell_id'],r['group']):r for r in previous['rows']}
    with threadpool_limits(limits=1):
        for fi,fold in enumerate(sorted({r['fold'] for r in old['manifest']})):
            name=fold.replace(':','__');_,_,test=sa.split(cells,fold)
            entries={r['group']:r for r in old['manifest'] if r['fold']==fold}
            sel=json.loads((candidate/'folds'/name/'selection.json').read_text())
            gain=sel['global_step']*sel['original_validation_selection']['weight']
            br=next(r for r in obj['manifest'] if r['fold']==fold)
            assert sa.digest(Path(br['branch_source']))==br['branch_sha256']
            branch=joblib.load(br['branch_source'])
            predictions={}
            for group,short in [('Base+M2','B2'),('Base+M1+M2','B12')]:
                entry=entries[group];path=FROZEN/entry['artifact']
                assert sa.digest(path)==entry['sha256'];model=joblib.load(path)
                assert not model.head.get('raw_reference',False)
                view=model.selection['reference_view'];index=model.selection['reference_index']
                iv=view.removesuffix('_consensus').removesuffix('_supported').removesuffix('_bagged').removesuffix('_soft')
                heads={}
                for seed in seeds:
                    h,counts=reweight(model.head,byid,seed+fi*10000)
                    heads[seed]=dict(h,views={iv:h['views'][iv]})
                    resamples.append(dict(fold=fold,group=group,seed=seed,counts=counts,source_ids=model.head['source_ids']))
                for c in test:
                    with np.load(SCREEN/name/'seed_0'/group/(c.id+'.npz')) as z:
                        base=z['parent'];physical=z['physical_control'];y=z['y']
                    _,res=sa.predict_components(model.parent,sa.prefix(c,sa.K),model.parent_mode)
                    for seed,h in heads.items():
                        e=rg.attach(dict(base=base,physical=physical,residual=res),sa.inference_view(c),h)
                        p=e['reference_predictions'][view][index]
                        predictions[seed,c.id,short]=p
                        metrics=sa.metrics(y,p);oldrow=prior[seed,c.id,group]
                        replay=max(replay,max(abs(metrics[k]-oldrow[k]) for k in ['mae','rmse','p95_ae']))
            for c in test:
                q=branch.predict(sa.inference_view(c))
                with np.load(candidate/'folds'/name/'predictions'/(c.id+'.npz')) as z:
                    controls={g:z[g].copy() for g in ['B','B1']};y=z['y']
                for seed in seeds:
                    pp=dict(controls,B2=predictions[seed,c.id,'B2'],B12=predictions[seed,c.id,'B12'])
                    pp.update({g+'3':p+gain*(q-p) for g,p in list(pp.items())})
                    dest=out/'folds'/name/str(seed)/(c.id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest,y=y,**pp)
                    for g,p in pp.items():
                        rows.append(dict(seed=seed,dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,p)))
            print(fold,'10 paired draws scored; v3 replay error',replay,flush=True)
    assert len(rows)==29200 and replay<1e-10
    table=aggregate(rows,['seed','dataset','group']);lookup={(r['seed'],r['dataset'],r['group']):r for r in table}
    gates=[]
    for seed in seeds:
        failures={a+'<'+b:[dict(dataset=ds,metric=m,delta_pp=lookup[seed,ds,a][m]-lookup[seed,ds,b][m])
                    for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae']
                    if lookup[seed,ds,a][m]>=lookup[seed,ds,b][m]] for a,b in EDGES}
        gates.append(dict(seed=seed,passed={k:not v for k,v in failures.items()},failures=failures))
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in old['manifest'])
    counts={a+'<'+b:sum(r['passed'][a+'<'+b] for r in gates) for a,b in EDGES}
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='COMPLETE',gates=gates,pass_counts=counts,table=table,
        original_v3_metric_replay_error=replay,resamples=resamples))
    protocol=json.loads((out/'protocol.json').read_text());protocol['status']='COMPLETE';sa.write_json(out/'protocol.json',protocol)
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
