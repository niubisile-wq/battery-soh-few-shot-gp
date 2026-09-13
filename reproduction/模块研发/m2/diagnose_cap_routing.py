"""Retrospective sequential cap/risk/selection decomposition; not deployable hybrids."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import repair_diagnostics as rd
import reference_gate as rg

STAGES=['v8','cap_only','new_risk_old_setting','v11']


def stage_predictions(e,cell,old_head,new_head,old_selection,new_selection):
    assert old_head['params']==new_head['params'] and old_head['settings']==new_head['settings']
    choices=[(old_head,old_selection),(dict(old_head,physical_cap=True),old_selection),
             (new_head,old_selection),(new_head,new_selection)]
    return {name:rg.attach(dict(e),cell,h)['reference_predictions'][s['reference_view']][s['reference_index']]
            for name,(h,s) in zip(STAGES,choices,strict=True)}


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['dataset'],r['group'],r['stage'],r['domain']].append(r)
    pooled=defaultdict(list)
    for (ds,group,stage,domain),rr in groups.items():
        pooled[ds,group,stage].append({k:float(np.mean([r[k] for r in rr])) for k in rd.KEYS})
    return [dict(dataset=ds,group=group,stage=stage,**{k:float(np.mean([r[k] for r in rr])) for k in rd.KEYS})
            for (ds,group,stage),rr in sorted(pooled.items())]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=rd.sa;root=sa.ROOT/'模块研发/results'
    oldroot=root/'m2_reference_candidate_v8';newroot=root/'m2_reference_candidate_v11'
    oldreq=json.loads((oldroot/'request.json').read_text());newreq=json.loads((newroot/'request.json').read_text())
    assert oldreq['status']==newreq['status']=='COMPLETE'
    old=json.loads((oldroot/'candidate.json').read_text());new=json.loads((newroot/'candidate.json').read_text())
    bynew={(m['fold'],m['group']):m for m in new['manifest']};cells=[c for ds in sa.PROTOCOL['datasets'] for c in sa.load_cells(ds)]
    rows=[];max_error=0.;settings=[]
    for item in old['manifest']:
        fold=item['fold'];group=item['group'];ni=bynew[fold,group]
        op=oldroot/item['artifact'];npth=newroot/ni['artifact']
        assert sa.digest(op)==item['sha256'] and sa.digest(npth)==ni['sha256']
        model=joblib.load(op);newmodel=joblib.load(npth)
        _,_,test=sa.split(cells,fold)
        settings.append(dict(fold=fold,group=group,old=model.selection,new=newmodel.selection))
        for c in test:
            relative=Path('folds')/fold.replace(':','__')/'seed_0'/group/(c.id+'.npz')
            with np.load(Path(oldreq['screen'])/relative) as z:
                base=z['parent'];physical=z['physical_control'];y=z['y'];old_pred=z[oldreq['variant']]
            with np.load(Path(newreq['screen'])/relative) as z:
                np.testing.assert_array_equal(y,z['y']);np.testing.assert_array_equal(base,z['parent']);np.testing.assert_array_equal(physical,z['physical_control']);new_pred=z[newreq['variant']]
            with np.load(Path(oldreq['screen'])/'folds'/fold.replace(':','__')/'seed_0'/'Base+M2'/(c.id+'.npz')) as z:raw=z['parent']
            support=sa.prefix(sa.inference_view(c),sa.K)
            _,res=sa.predict_components(model.parent,support,model.parent_mode)
            _,rawres=sa.predict_components(model.raw_model,support,model.raw_mode)
            e=dict(base=base,physical=physical,residual=res,raw_base=raw,raw_residual=rawres)
            predictions=stage_predictions(e,sa.inference_view(c),model.head,newmodel.head,model.selection,newmodel.selection)
            for stage,expected in [('v8',old_pred),('v11',new_pred)]:
                error=float(np.max(abs(predictions[stage]-expected)));assert error<1e-10;max_error=max(max_error,error)
            for stage,p in predictions.items():
                m=sa.metrics(y,p);rows.append(dict(fold=fold,dataset=c.dataset,domain=c.domain,cell_id=c.id,group=group,stage=stage,**{k:100*float(m[k]) for k in rd.KEYS}))
        print(fold,group,'replayed',flush=True)
    summary=summarize(rows)
    sa.write_json(out/'result.json',dict(status='COMPLETE',summary=summary,rows=rows,settings=settings,max_endpoint_replay_error=max_error,
        input_hashes={str(p):sa.digest(p) for p in [oldroot/'candidate.json',newroot/'candidate.json']},
        limits=['Retrospective explored development tests; hybrids are mechanism diagnostics, not deployable candidates.',
                'cap_only uses old uncapped source risks; new_risk_old_setting keeps previous validation setting. Neither was independently selected.',
                'Sequential metric differences depend on path and are not independent causal contributions. No threshold tuning here.']))
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    print(json.dumps([r for r in summary if r['group']=='Base+M1+M2'],indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
