"""All152 budget-restricted inner domains, fit+refit audit, never outer-score."""
import csv,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from source_oof import sa,HERE


def checked(root,fold,held,protocol,reused=False):
    req=json.loads((root/'request.json').read_text());r=json.loads((root/'result.json').read_text())
    audit=json.loads((root/'verification.json').read_text())
    assert req['fold']==fold and req['held']==held and req['protocol']==protocol
    assert audit['status'] in ('PASS','REPLAY_PASS_OPTIMIZER_FAILURE_PRESENT')
    assert audit['result_sha256']==sa.digest(root/'result.json') and audit['request_sha256']==sa.digest(root/'request.json')
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    assert all(sa.digest(HERE/n)==h for n,h in audit['helper_hashes'].items())
    assert max(audit['errors'].values())<1e-8
    if not reused:assert audit['code_sha256']==sa.digest(HERE/'audit_source_domain_pilot.py')
    failures=[]
    for spec in r['models']:
        assert sa.digest(Path(spec['path']))==spec['sha256']
        if not spec['optimization']['success']:failures.append(dict(kernel=spec['kernel'],optimization=spec['optimization']))
    return dict(fold=fold,held=held,root=str(root),status='REPLAY_AUDITED',optimizer_failures=failures,reused=reused,
        result_sha256=sa.digest(root/'result.json'),audit_sha256=sa.digest(root/'verification.json'),audit_code_sha256=audit['code_sha256'])


def run(fold,held,out,protocol):
    folder=out/'tasks'/(fold.replace(':','__')+'___'+held);branch=folder/'branch'
    try:
        folder.mkdir(parents=True,exist_ok=False)
        for script,args in [('source_domain_pilot.py',['--fold',fold,'--held',held,'--out',str(branch)]),
                            ('audit_source_domain_pilot.py',['--root',str(branch)])]:
            with (folder/(script+'.log')).open('w') as log:
                p=subprocess.run([sys.executable,str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:return dict(fold=fold,held=held,status='TASK_FAILED',script=script,exit_code=p.returncode,root=str(branch))
        return checked(branch,fold,held,protocol)
    except Exception as exc:
        return dict(fold=fold,held=held,status='TASK_FAILED',error=repr(exc),root=str(branch))


def main():
    results=HERE.parent/'results';feasibility=results/'m4_source_domain_feasibility_v1'
    summary=json.loads((feasibility/'summary.json').read_text());assert summary['status']=='BUDGET_AND_CACHE_COVERAGE_VERIFIED'
    with (feasibility/'folds.csv').open() as f:rows=list(csv.DictReader(f))
    tasks=sorted((r['fold'],r['held_source_domain']) for r in rows)
    assert len(tasks)==len(set(tasks))==summary['inner_domain_tasks']==152
    protocol=json.loads((HERE/'source_domain_protocol.json').read_text())
    selection=results/'m4_source_domain_matr_b1_selection_v1'
    selection_audit=json.loads((selection/'verification.json').read_text())
    assert selection_audit['status']=='AGGREGATION_PASS_OPTIMIZER_FAILURE_RETAINED'
    assert selection_audit['summary_sha256']==sa.digest(selection/'summary.json')
    reuse={('MATR:b1','b2'):results/'m4_source_domain_pilot_v1',
           **{('MATR:b1',d):results/'m4_source_domain_matr_b1_extension_v1'/d for d in ('b3','b4')}}
    done=[checked(root,*key,protocol,reused=True) for key,root in reuse.items()]
    out=results/'m4_source_domain_batch_v1';out.mkdir(exist_ok=False)
    req=json.loads((reuse['MATR:b1','b3']/'request.json').read_text())
    sa.write_json(out/'request.json',dict(tasks=[dict(fold=f,held=h) for f,h in tasks],workers=3,protocol=protocol,
        feasibility_sha256=sa.digest(feasibility/'summary.json'),coverage_sha256=sa.digest(feasibility/'folds.csv'),reused=done,
        pilot_selection_audit_sha256=sa.digest(selection/'verification.json'),
        code_hashes={**req['code_hashes'],**{n:sa.digest(HERE/n) for n in ('batch_source_domain.py','audit_source_domain_pilot.py','audit_support_cv.py','audit_conditional_mixed.py')}},
        limits='No automatic outer scoring. Retain optimizer warnings,do not exclude/restart by outcome. Three prior audits retain historical code hashes; audit argument support changed,not its arithmetic.'))
    sa.write_json(out/'progress.json',dict(completed=done,total=152))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,f,h,out,protocol) for f,h in tasks if (f,h) not in reuse]
        for future in as_completed(futures):
            r=future.result();done.append(r)
            sa.write_json(out/'progress.json',dict(completed=done,total=152));print(r,flush=True)
    assert len(done)==152
    sa.write_json(out/'result.json',dict(status='ALL_INNER_TASKS_REPLAY_AUDITED' if all(r['status']=='REPLAY_AUDITED' for r in done) else 'BATCH_HAS_TASK_FAILURES',
        tasks=done,optimizer_failure_count=sum(len(r.get('optimizer_failures',[])) for r in done)))


if __name__=='__main__':main()
