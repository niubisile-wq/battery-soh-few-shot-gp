"""Independent weighted aggregation of audited pilot source-domain losses."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from source_oof import sa,HERE


def main():
    root=HERE.parent/'results/m4_source_domain_matr_b1_selection_v1'
    summary=json.loads((root/'summary.json').read_text());buckets=defaultdict(list);seen=set();failures=[]
    assert len(summary['provenance'])==3
    for origin in summary['provenance']:
        folder=Path(origin['root'])
        for file,key in [('verification.json','audit_sha256'),('result.json','result_sha256'),('request.json','request_sha256')]:
            assert sa.digest(folder/file)==origin[key]
        audit=json.loads((folder/'verification.json').read_text());result=json.loads((folder/'result.json').read_text())
        assert audit['status'] in ('PASS','REPLAY_PASS_OPTIMIZER_FAILURE_PRESENT')
        for spec in result['models']:
            if not spec['optimization']['success']:failures.append((origin['held'],spec['kernel']))
        for r in result['rows']:
            key=(r['cell_id'],r['domain'],r['key'],r['group'],r['gain']);assert key not in seen;seen.add(key)
            if r['group']=='B123':buckets[r['key'],r['gain']].append(r)
    assert len(seen)==summary['rows']==9600 and len(buckets)==20
    trials=[];error=0.
    for t in summary['trials']:
        rr=buckets[t['key'],t['gain']];counts={d:sum(r['domain']==d for r in rr) for d in ('b2','b3','b4')}
        assert all(counts.values())
        value=sum(float(r['mae'])/(3*counts[r['domain']]) for r in rr)
        error=max(error,abs(value-t['mae']));trials.append(dict(t,mae=value))
    assert error<1e-10
    for family,s in summary['selected'].items():
        best=min([r for r in trials if r['family']==family],key=lambda r:(r['mae'],r['gain'],r['key']))
        assert (best['key'],best['gain'])==(s['key'],s['gain'])
        kernel=s['key'].split('__')[1]
        assert sorted(s['selected_kernel_optimizer_failures'])==sorted(d for d,k in failures if k==kernel)
    assert sorted(failures)==sorted((r['held'],r['kernel']) for r in summary['optimizer_failures'])
    sa.write_json(root/'verification.json',dict(status='AGGREGATION_PASS_OPTIMIZER_FAILURE_RETAINED',rows=9600,trials=20,
        error=error,summary_sha256=sa.digest(root/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        limits='Independent weighted aggregation/selection from previously audited cell losses; no new GP fits or outer query evaluation.'))
    print('PASS source aggregation, optimizer failures retained',error,flush=True)


if __name__=='__main__':main()
