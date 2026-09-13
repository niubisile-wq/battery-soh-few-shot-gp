"""Source-budget feasibility for within-cell observed-label difference learning."""
import json
from pathlib import Path
import numpy as np
from source_oof import sa,HERE


def main():
    results=HERE.parent/'results';out=results/'m4_increment_pair_feasibility_v1';out.mkdir(exist_ok=False)
    source=json.loads((results/'m4_source_batch_v1/result.json').read_text())
    byds={ds:{c.id:c for c in sa.load_cells(ds)} for ds in ('XJTU','MATR','Tongji')}
    rows=[]
    for fold in source['folds']:
        req=json.loads((Path(fold['source'])/'request.json').read_text());allowed={}
        for cid,j in req['source_keys']:allowed.setdefault(cid,[]).append(j)
        cells=byds[fold['fold'].split(':')[0]];pairs=[];gaps=[];changes=[];query_pairs=0
        for cid,indices in allowed.items():
            c=cells[cid];ix=sorted(set(indices));assert len(ix)==len(indices)
            for a,b in zip(ix[:-1],ix[1:]):
                assert np.isfinite(c.y[[a,b]]).all()
                pairs.append((cid,a,b));gaps.append(float(c.cycle[b]-c.cycle[a]));changes.append(float(c.y[b]-c.y[a]))
                query_pairs+=b>=sa.K
        assert len(pairs)==len(req['source_keys'])-len(allowed)
        assert all((cid,a) in {tuple(k) for k in req['source_keys']} and (cid,b) in {tuple(k) for k in req['source_keys']} for cid,a,b in pairs)
        rows.append(dict(fold=fold['fold'],source_labels=len(req['source_keys']),source_cells=len(allowed),
            consecutive_budget_pairs=len(pairs),pairs_ending_after_support=query_pairs,
            median_cycle_gap=float(np.median(gaps)),max_cycle_gap=float(max(gaps)),
            median_abs_delta_soh_pp=float(np.median(abs(np.array(changes)))*100),
            fraction_positive_delta=float(np.mean(np.array(changes)>0)),
            provenance_sha256=sa.digest(Path(fold['source'])/'request.json')))
        sa.write_json(out/(fold['fold'].replace(':','__')+'.json'),dict(fold=fold['fold'],pairs=pairs,
            limits='Pairs share endpoints; do not count them as independent labels. Gaps vary; not consecutive physical cycles. No out-of-budget intermediate labels.'))
    sa.write_json(out/'summary.json',dict(status='SOURCE_PAIR_FEASIBILITY_COMPLETE',folds=rows,
        code_sha256=sa.digest(Path(__file__)),
        limits='Descriptive source-only feasibility,not fitted model or performance evidence. Any difference observation model must account for shared-endpoint noise and target anchoring.'))
    for r in rows:
        print(r['fold'],r['consecutive_budget_pairs'],r['pairs_ending_after_support'],r['median_cycle_gap'],r['max_cycle_gap'],flush=True)


if __name__=='__main__':main()
