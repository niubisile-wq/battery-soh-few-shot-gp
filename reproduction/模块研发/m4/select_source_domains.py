"""All21 outer selections, only after all152 source-domain tasks are audited."""
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa,HERE
from batch_source_domain import checked
from evaluate import dump_csv
from score import PARENTS


def main():
    results=HERE.parent/'results';batch=results/'m4_source_domain_batch_v1'
    request=json.loads((batch/'request.json').read_text());result=json.loads((batch/'result.json').read_text())
    assert result['status']=='ALL_INNER_TASKS_REPLAY_AUDITED'
    assert all(sa.digest(HERE/n)==h for n,h in request['code_hashes'].items())
    expected={(r['fold'],r['held']) for r in request['tasks']};actual={(r['fold'],r['held']) for r in result['tasks']}
    assert expected==actual and len(expected)==len(result['tasks'])==152
    protocol=request['protocol'];grouped=defaultdict(list)
    for task in result['tasks']:grouped[task['fold']].append(task)
    assert len(grouped)==21
    fullsource=results/'m4_support_cv_validation_v1';sourceaudit=json.loads((fullsource/'verification.json').read_text())
    assert sourceaudit['status']=='PASS' and sourceaudit['result_sha256']==sa.digest(fullsource/'result.json')
    source_records={r['fold']:r for r in json.loads((fullsource/'result.json').read_text())['selections']}
    assert set(source_records)==set(grouped)
    selections=[];table=[];models=[];total_rows=0
    for fold,tasks in sorted(grouped.items()):
        domains={t['held'] for t in tasks};buckets=defaultdict(lambda:defaultdict(list));origins=[]
        for task in tasks:
            root=Path(task['root']);record=checked(root,fold,task['held'],protocol,reused=task['reused'])
            assert record==task
            r=json.loads((root/'result.json').read_text())
            ids={row['cell_id'] for row in r['rows']};keys={f+'__'+k for f in protocol['families'] for k in protocol['kernels']}
            rowkeys={(cid,k,g,a) for cid in ids for k in keys for g in PARENTS for a in protocol['gains']}
            assert len(r['rows'])==len(rowkeys) and {(v['cell_id'],v['key'],v['group'],v['gain']) for v in r['rows']}==rowkeys
            for row in r['rows']:
                assert row['domain']==task['held']
                buckets[row['key'],row['gain'],row['group']][task['held']].append(float(row['mae']));total_rows+=1
            for spec in r['models']:
                gp=joblib.load(spec['path']).gp;theta,bounds=gp.kernel_.theta,gp.kernel_.bounds
                hits=[int(i) for i in range(len(theta)) if min(abs(theta[i]-bounds[i]))<1e-4]
                models.append(dict(fold=fold,held=task['held'],kernel=spec['kernel'],success=spec['optimization']['success'],
                    message=spec['optimization']['message'],iterations=spec['optimization']['iterations'],rho=gp.rho_,
                    rho_bound_hit=gp.diagnostics_['rho_bound_hit'],kernel_bound_coordinates=json.dumps(hits),
                    model_sha256=spec['sha256']))
            origins.append(task)
        local=[]
        for (key,gain,g),losses in sorted(buckets.items()):
            assert set(losses)==domains
            local.append(dict(fold=fold,key=key,target=key.split('__')[0],gain=gain,group=g,
                source_validation_mae=float(np.mean([np.mean(v) for v in losses.values()])),
                domains=len(domains),cells=sum(map(len,losses.values()))))
        assert len(local)==160;table.extend(local)
        trials=[r for r in local if r['group']=='B123'];choices={}
        outerpath=fullsource/'folds'/fold.replace(':','__')/'result.json';outer=json.loads(outerpath.read_text())
        assert outerpath==Path(source_records[fold]['root'])/'result.json'
        assert sa.digest(outerpath)==source_records[fold]['result_sha256']
        for family in protocol['families']:
            choice=min([r for r in trials if r['target']==family],key=lambda r:(r['source_validation_mae'],r['gain'],r['key']))
            kernel=choice['key'].split('__')[1];spec=next(m for m in outer['models'] if m['key']=='mixed__'+kernel)
            assert sa.digest(Path(spec['path']))==spec['sha256']
            warnings=[m for m in models if m['fold']==fold and m['kernel']==kernel and not m['success']]
            choices[family]=dict(selected=choice,model=spec,inner_optimizer_failures=warnings)
        selections.append(dict(fold=fold,choices=choices,trials=trials,inner_tasks=origins,
            full_source_record=str(outerpath),full_source_record_sha256=sa.digest(outerpath)))
        print(fold,{f:(c['selected']['key'],c['selected']['gain'],len(c['inner_optimizer_failures'])) for f,c in choices.items()},flush=True)
    assert len(models)==304 and len(table)==3360
    out=results/'m4_source_domain_selection_v1';out.mkdir(exist_ok=False)
    dump_csv(out/'comparison.csv',table);dump_csv(out/'models.csv',models)
    sa.write_json(out/'summary.json',dict(status='ALL_OUTER_SOURCE_CHOICES_AUDIT_PENDING',selections=selections,rows=total_rows,
        tasks=152,models=304,optimizer_failures=[m for m in models if not m['success']],
        kernel_bound_hits=[m for m in models if m['kernel_bound_coordinates']!='[]'],rho_bound_hits=[m for m in models if m['rho_bound_hit']],
        protocol=protocol,batch_result_sha256=sa.digest(batch/'result.json'),batch_request_sha256=sa.digest(batch/'request.json'),
        full_source_audit_sha256=sa.digest(fullsource/'verification.json'),code_sha256=sa.digest(Path(__file__)),
        limits='No outer query scoring; every inner task retained including optimizer failures. Source errors are sparse budget-only. Historical parent configurations inherited. Source choice may fully replace parents at gain1; no automatic adoption.'))


if __name__=='__main__':main()
