"""Aggregate all three MATR:b1 source domains, retaining failed optimizer flags."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa,HERE
from evaluate import dump_csv
from score import PARENTS


def main():
    results=HERE.parent/'results';protocol=json.loads((HERE/'source_domain_protocol.json').read_text())
    roots=[results/'m4_source_domain_pilot_v1']+[results/'m4_source_domain_matr_b1_extension_v1'/d for d in ('b3','b4')]
    buckets=defaultdict(lambda:defaultdict(list));provenance=[];models=[];domains=set();count=0
    for root in roots:
        req=json.loads((root/'request.json').read_text());r=json.loads((root/'result.json').read_text())
        audit=json.loads((root/'verification.json').read_text())
        assert req['fold']=='MATR:b1' and req['protocol']==protocol
        assert req['held'] not in domains;domains.add(req['held'])
        assert audit['status'] in ('PASS','REPLAY_PASS_OPTIMIZER_FAILURE_PRESENT')
        assert audit['result_sha256']==sa.digest(root/'result.json') and audit['request_sha256']==sa.digest(root/'request.json')
        assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
        assert all(sa.digest(HERE/n)==h for n,h in audit['helper_hashes'].items())
        assert max(audit['errors'].values())<1e-8
        for spec in r['models']:
            assert sa.digest(Path(spec['path']))==spec['sha256']
            models.append(dict(held=req['held'],**spec))
        keys={f+'__'+k for f in protocol['families'] for k in protocol['kernels']}
        ids={v['cell_id'] for v in r['rows']}
        expected={(cid,k,g,a) for cid in ids for k in keys for g in PARENTS for a in protocol['gains']}
        assert len(r['rows'])==len(expected) and {(v['cell_id'],v['key'],v['group'],v['gain']) for v in r['rows']}==expected
        for row in r['rows']:
            assert row['domain']==req['held']
            buckets[row['key'],row['gain'],row['group']][req['held']].append(row['mae']);count+=1
        provenance.append(dict(root=str(root),held=req['held'],audit_status=audit['status'],
            audit_sha256=sa.digest(root/'verification.json'),audit_code_sha256=audit['code_sha256'],
            result_sha256=sa.digest(root/'result.json'),request_sha256=sa.digest(root/'request.json')))
    assert domains=={'b2','b3','b4'} and len(models)==6
    table=[]
    for (key,gain,group),losses in sorted(buckets.items()):
        assert set(losses)==domains
        table.append(dict(key=key,family=key.split('__')[0],gain=gain,group=group,
            mae=float(np.mean([np.mean(v) for v in losses.values()])),domains=3,cells=sum(map(len,losses.values()))))
    assert len(table)==160
    trials=[r for r in table if r['group']=='B123'];assert len(trials)==20
    selected={f:dict(min([r for r in trials if r['family']==f],key=lambda r:(r['mae'],r['gain'],r['key']))) for f in protocol['families']}
    for f,s in selected.items():
        kernel=s['key'].split('__')[1]
        s['selected_kernel_optimizer_failures']=[m['held'] for m in models if m['kernel']==kernel and not m['optimization']['success']]
    oldpath=results/'m4_support_cv_validation_v1/folds/MATR__b1/result.json'
    original=json.loads(oldpath.read_text())['selected']
    out=results/'m4_source_domain_matr_b1_selection_v1';out.mkdir(exist_ok=False)
    dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SINGLE_OUTER_SOURCE_SELECTION_NOT_ADOPTED',fold='MATR:b1',rows=count,
        selected=selected,original_validation_selected=original,trials=trials,
        optimizer_failures=[m for m in models if not m['optimization']['success']],provenance=provenance,
        code_sha256=sa.digest(Path(__file__)),protocol_sha256=sa.digest(HERE/'source_domain_protocol.json'),
        old_selection_sha256=sa.digest(oldpath),
        limits='All three source domains,all six models included,no optimizer failure exclusions. Old selection read only for comparison after source selection. No outer validation/test metrics scored; inherited parent choices remain.'))
    print('selected',selected,'all optimizer failures',[(m['held'],m['kernel']) for m in models if not m['optimization']['success']],flush=True)


if __name__=='__main__':main()
