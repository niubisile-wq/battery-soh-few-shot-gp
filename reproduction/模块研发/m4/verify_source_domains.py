"""Independent all21 source-domain aggregation, choices and warning coverage."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa,HERE


def main():
    results=HERE.parent/'results';root=results/'m4_source_domain_selection_v1';batch=results/'m4_source_domain_batch_v1'
    s=json.loads((root/'summary.json').read_text())
    assert s['status']=='ALL_OUTER_SOURCE_CHOICES_AUDIT_PENDING'
    assert s['batch_result_sha256']==sa.digest(batch/'result.json') and s['batch_request_sha256']==sa.digest(batch/'request.json')
    assert s['code_sha256']==sa.digest(HERE/'select_source_domains.py')
    request=json.loads((batch/'request.json').read_text());expected={(r['fold'],r['held']) for r in request['tasks']}
    with (root/'comparison.csv').open() as f:table=list(csv.DictReader(f))
    saved={(r['fold'],r['key'],float(r['gain']),r['group']):r for r in table}
    assert len(saved)==len(table)==3360 and len(s['selections'])==21
    seen=set();failures=[];rows=0;error=0.;checked_models=0
    for selection in s['selections']:
        fold=selection['fold'];buckets=defaultdict(list);domains=set()
        for task in selection['inner_tasks']:
            taskkey=(fold,task['held']);assert taskkey not in seen;seen.add(taskkey);domains.add(task['held'])
            folder=Path(task['root'])
            assert sa.digest(folder/'result.json')==task['result_sha256'] and sa.digest(folder/'verification.json')==task['audit_sha256']
            r=json.loads((folder/'result.json').read_text());a=json.loads((folder/'verification.json').read_text())
            assert a['status'] in ('PASS','REPLAY_PASS_OPTIMIZER_FAILURE_PRESENT') and a['result_sha256']==task['result_sha256']
            for model in r['models']:
                assert sa.digest(Path(model['path']))==model['sha256'];checked_models+=1
                if not model['optimization']['success']:failures.append((fold,task['held'],model['kernel']))
            for v in r['rows']:buckets[v['key'],float(v['gain']),v['group']].append(v);rows+=1
        computed={}
        for (key,gain,group),rr in buckets.items():
            counts={d:sum(v['domain']==d for v in rr) for d in domains};assert all(counts.values())
            value=sum(float(v['mae'])/(len(domains)*counts[v['domain']]) for v in rr)
            old=saved[fold,key,gain,group];assert int(old['domains'])==len(domains) and int(old['cells'])==len(rr)
            error=max(error,abs(value-float(old['source_validation_mae'])));computed[key,gain,group]=value
        assert len(computed)==160
        assert len(selection['trials'])==20
        for t in selection['trials']:error=max(error,abs(t['source_validation_mae']-computed[t['key'],float(t['gain']),'B123']))
        for family,c in selection['choices'].items():
            best=min([(k,a) for k,a,g in computed if g=='B123' and k.startswith(family+'__')],
                key=lambda item:(computed[item[0],item[1],'B123'],item[1],item[0]))
            assert best==(c['selected']['key'],float(c['selected']['gain']))
            warnings={(r['fold'],r['held'],r['kernel']) for r in c['inner_optimizer_failures']}
            assert warnings=={r for r in failures if r[0]==fold and r[2]==best[0].split('__')[1]}
            path=Path(selection['full_source_record']);assert sa.digest(path)==selection['full_source_record_sha256']
            full=json.loads(path.read_text());assert c['model'] in full['models']
            assert sa.digest(Path(c['model']['path']))==c['model']['sha256']
        print(fold,'independent all-source aggregation',flush=True)
    assert seen==expected and len(seen)==152 and checked_models==304 and rows==s['rows'] and error<1e-10
    assert set(failures)=={(m['fold'],m['held'],m['kernel']) for m in s['optimizer_failures']}
    sa.write_json(root/'verification.json',dict(status='AGGREGATION_PASS_WARNINGS_RETAINED',rows=rows,group_configs=3360,
        objectives=420,source_models=304,optimizer_failures=len(failures),error=error,
        summary_sha256=sa.digest(root/'summary.json'),comparison_sha256=sa.digest(root/'comparison.csv'),code_sha256=sa.digest(Path(__file__)),
        limits='Independent weighted means and choices from audited source losses; no new outer query data. Reproduced optimizer failures not relabeled as converged.'))
    print('PASS all21source selections',error,'warnings',len(failures),flush=True)


if __name__=='__main__':main()
