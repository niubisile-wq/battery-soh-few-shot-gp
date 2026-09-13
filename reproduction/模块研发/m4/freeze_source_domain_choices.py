"""Freeze source-domain selected and original-validation control configurations.

No raw cells or query predictions are read here. This must complete before
the next outer development screen. It refuses incomplete inner batches.
"""
import json
from pathlib import Path
from source_oof import sa,HERE,FROZEN


def build_manifest(results):
    selection=results/'m4_source_domain_selection_v1';batch=results/'m4_source_domain_batch_v1'
    s=json.loads((selection/'summary.json').read_text());audit=json.loads((selection/'verification.json').read_text())
    b=json.loads((batch/'result.json').read_text());req=json.loads((batch/'request.json').read_text())
    assert b['status']=='ALL_INNER_TASKS_REPLAY_AUDITED' and len(b['tasks'])==152
    assert audit['status']=='AGGREGATION_PASS_WARNINGS_RETAINED'
    assert audit['summary_sha256']==sa.digest(selection/'summary.json')
    assert audit['comparison_sha256']==sa.digest(selection/'comparison.csv')
    assert audit['code_sha256']==sa.digest(HERE/'verify_source_domains.py')
    assert s['code_sha256']==sa.digest(HERE/'select_source_domains.py')
    assert s['batch_result_sha256']==sa.digest(batch/'result.json') and s['batch_request_sha256']==sa.digest(batch/'request.json')
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    expected={(r['fold'],r['held']) for r in req['tasks']}
    assert expected=={(r['fold'],r['held']) for r in b['tasks']} and len(expected)==152
    control=results/'m4_support_cv_screen_v1'
    ca=json.loads((control/'verification.json').read_text());cf=json.loads((control/'frozen_selections.json').read_text())
    assert ca['status']=='PASS' and ca['summary_sha256']==sa.digest(control/'summary.json')
    controls={r['fold']:r for r in cf['selections']}
    assert set(controls)=={r['fold'] for r in s['selections']} and len(controls)==len(s['selections'])==21
    selections=[];taskkeys=set()
    for fold in s['selections']:
        f=fold['fold'];choices={}
        record=Path(fold['full_source_record'])
        assert sa.digest(record)==fold['full_source_record_sha256']
        full=json.loads(record.read_text())
        for task in fold['inner_tasks']:
            key=(f,task['held']);assert key not in taskkeys;taskkeys.add(key)
            root=Path(task['root'])
            assert sa.digest(root/'result.json')==task['result_sha256'] and sa.digest(root/'verification.json')==task['audit_sha256']
        for family,c in fold['choices'].items():
            chosen=c['selected']
            best=min([r for r in fold['trials'] if r['target']==family],key=lambda r:(r['source_validation_mae'],r['gain'],r['key']))
            assert chosen==best and c['model'] in full['models']
            assert c['model']['key']=='mixed__'+chosen['key'].split('__')[1]
            assert sa.digest(Path(c['model']['path']))==c['model']['sha256']
            choices[family]=c
        for family in ('cv','uniform'):
            c=controls[f]['choices'][family]
            assert sa.digest(Path(c['model']['path']))==c['model']['sha256']
            assert c['model'] in full['models']
            choices['original_'+family]=c
        assert set(choices)=={'cv','uniform','original_cv','original_uniform'}
        selections.append(dict(fold=f,choices=choices,full_source_record=str(record),full_source_record_sha256=sa.digest(record)))
    assert taskkeys==expected
    parent=results/'m3_waveform_candidate_v1'
    assert cf['parent_manifest_sha256']==sa.digest(parent/'candidate.json')
    assert cf['m2_manifest_sha256']==sa.digest(FROZEN/'candidate.json')
    for folder in (parent,FROZEN):
        manifest=json.loads((folder/'candidate.json').read_text())
        assert all(sa.digest(folder/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    return dict(status='ALL_CHOICES_FROZEN_NO_NEW_QUERY_SCORES',selections=selections,protocol=s['protocol'],
        source_selection_sha256=sa.digest(selection/'summary.json'),source_selection_audit_sha256=sa.digest(selection/'verification.json'),
        original_control_summary_sha256=sa.digest(control/'summary.json'),original_control_frozen_sha256=sa.digest(control/'frozen_selections.json'),
        parent_manifest_sha256=sa.digest(parent/'candidate.json'),m2_manifest_sha256=sa.digest(FROZEN/'candidate.json'),
        code_sha256=sa.digest(Path(__file__)),
        limits='42 source-selected and42 original-validation control choices. Optimizer warnings retained,not convergence certification. No raw/query data read by this freeze. Repeated outer development cohorts,not independent confirmation.')


def main():
    results=HERE.parent/'results';manifest=build_manifest(results)
    out=results/'m4_source_domain_screen_v1';out.mkdir(exist_ok=False)
    sa.write_json(out/'frozen_selections.json',manifest)
    print('21folds x4families frozen,no query scoring',flush=True)


if __name__=='__main__':main()
