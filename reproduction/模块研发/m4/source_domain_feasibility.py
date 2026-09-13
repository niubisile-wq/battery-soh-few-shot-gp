"""Audit original-budget source-domain validation coverage; no new scoring."""
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa,HERE
from score import PARENTS
from evaluate import dump_csv


def main():
    results=HERE.parent/'results';batchpath=results/'m4_source_batch_v1/result.json'
    batch=json.loads(batchpath.read_text());assert batch['status']=='VALIDATION_BATCH_COMPLETE' and len(batch['folds'])==21
    rows=[];provenance=[];fit_count=0
    for entry in batch['folds']:
        root=Path(entry['source']);request=json.loads((root/'request.json').read_text())
        manifest=json.loads((root/'audit_manifest.json').read_text());audit=json.loads((root/'verification.json').read_text())
        assert audit['status']=='PASS' and audit['code_sha256']==sa.digest(HERE/'audit_source_oof.py')
        assert request['code_sha256']==sa.digest(HERE/'source_oof.py')
        fold=entry['fold'];assert request['fold']==manifest['fold']==fold
        provpath=results/'m2_strict_ablation_v1/folds'/fold.replace(':','__')/'provenance.json'
        original=json.loads(provpath.read_text());budget=set(map(tuple,request['source_keys']))
        assert budget==set(map(tuple,original['source_keys'])) and len(budget)<=1000
        train,val,test=sa.split(sa.load_cells(fold.split(':')[0]),fold);byid={c.id:c for c in train}
        assert {cid for cid,j in budget}==set(byid) and set(byid).isdisjoint({c.id for c in val+test})
        allowed=defaultdict(list)
        for cid,j in sorted(budget):allowed[cid].append(j)
        path=root/'complete_parent_source_oof.joblib';assert sa.digest(path)==manifest['output_sha256']
        episodes=joblib.load(path);assert len(episodes)==len(train)==audit['source_cells']
        assert {e['cell_id'] for e in episodes}==set(byid)
        domains=sorted({c.domain for c in train})
        for spec in manifest['fits']:
            assert sa.digest(root/spec['artifact'])==spec['sha256']
            expected={k for k in budget if byid[k[0]].domain not in spec['excluded']}
            assert set(map(tuple,spec['fit_keys']))==expected
        for head in manifest['heads']:
            assert sa.digest(root/head['artifact'])==head['sha256']
            assert set(head['train_ids'])=={c.id for c in train if c.domain!=head['held']}
        fit_count+=len(manifest['fits'])
        for held in domains:
            fit_keys={k for k in budget if byid[k[0]].domain!=held}
            held_keys=budget-fit_keys;ee=[e for e in episodes if e['domain']==held]
            counts=[];gaps=[]
            for e in ee:
                c=byid[e['cell_id']];q=np.asarray(e['query_indices'],int)
                assert c.domain==held and q.tolist()==[j for j in allowed[c.id] if j>=sa.K]
                assert len(q)>0 and set(range(sa.K))<=set(allowed[c.id])
                assert np.array_equal(e['y'],c.y[q]) and set(e['predictions'])==set(PARENTS)
                assert all(np.asarray(p).shape==q.shape and np.isfinite(p).all() for p in e['predictions'].values())
                counts.append(len(q));gaps.extend(np.diff(np.r_[sa.K-1,q]).tolist())
            assert fit_keys.isdisjoint(held_keys) and fit_keys|held_keys==budget
            rows.append(dict(fold=fold,dataset=fold.split(':')[0],held_source_domain=held,
                train_domains=len(domains)-1,train_cells=len({cid for cid,j in fit_keys}),train_labels=len(fit_keys),
                held_cells=len(ee),held_support_labels=sa.K*len(ee),held_query_labels=sum(counts),
                query_labels_per_cell_min=min(counts),query_labels_per_cell_max=max(counts),
                query_index_gap_max=max(gaps),parent_groups=8))
        provenance.append(dict(fold=fold,root=str(root),budget_sha256=sa.digest(provpath),
            request_sha256=sa.digest(root/'request.json'),manifest_sha256=sa.digest(root/'audit_manifest.json'),
            audit_sha256=sa.digest(root/'verification.json'),cache_sha256=sa.digest(path)))
        print(fold,'source budget/OOF coverage verified',flush=True)
    out=results/'m4_source_domain_feasibility_v1';out.mkdir(exist_ok=False)
    dump_csv(out/'folds.csv',rows)
    summary=dict(status='BUDGET_AND_CACHE_COVERAGE_VERIFIED',outer_folds=21,inner_domain_tasks=len(rows),
        checked_existing_source_models=fit_count,existing_parent_retraining_required=False,
        source_encoder_inner_refit_required=True,optimized_new_branch_fits_two_kernels=2*len(rows),
        provisional_nonoptimized_encoder_reference_fits=2*len(rows),
        min_fit_labels=min(r['train_labels'] for r in rows),min_held_query_labels=min(r['held_query_labels'] for r in rows),
        min_queries_per_cell=min(r['query_labels_per_cell_min'] for r in rows),provenance=provenance,
        batch_sha256=sa.digest(batchpath),code_sha256=sa.digest(Path(__file__)),
        limits=['Coverage/integrity checks,not a fresh rerun of all parent predictions',
            'Sparse original-budget query labels,not complete held-cell trajectories',
            'Outer-frozen historical parent choices inherited; not fully nested architecture selection',
            'Full-source Haar/PCA/scalers cannot be reused in inner branch fits because they include held-source inputs',
            'Repeated development outer cohorts; no independent confirmation,no new candidate selected'])
    sa.write_json(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k not in ('provenance','limits')},flush=True)


if __name__=='__main__':main()
