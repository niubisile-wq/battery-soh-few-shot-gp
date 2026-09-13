"""Bind audited evidence for the experimental M3 release, without changing models."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from screen import sa,HERE,FROZEN


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=HERE.parent/'results';candidate=root/'m3_waveform_candidate_v1'
    paths={'candidate':candidate/'candidate.json',
           'stability':root/'m3_waveform_stability_v2/summary.json',
           'prediction_audit':root/'m3_waveform_audit_v1/audit.json',
           'training_audit':root/'m3_waveform_training_audit_v1/audit.json',
           'controls':root/'m3_waveform_screen_v1/matched_controls/summary.json',
           'sensitivity':root/'m3_waveform_shrink_v1/summary.json'}
    objects={k:json.loads(p.read_text()) for k,p in paths.items()};obj=objects['candidate']
    assert objects['stability']['status']=='COMPLETE'
    assert objects['prediction_audit']['status']==objects['training_audit']['status']=='PASS'
    assert len(objects['training_audit']['folds'])==21
    assert objects['prediction_audit']['audited_rows']==29200
    assert max(obj[k] for k in ['max_cache_error','max_serialized_replay_error','max_boundary_error'])<1e-8
    assert len(obj['manifest'])==84 and len(obj['data_hashes'])==365
    for r in obj['manifest']:assert sa.digest(candidate/r['artifact'])==r['sha256']
    for relative,digest in obj['source_snapshot'].items():
        path=candidate/relative;assert sa.digest(path)==digest
        original=HERE.parent/Path(relative).relative_to('source_snapshot')
        assert sa.digest(original)==digest,original
    for r in obj['data_hashes'].values():assert sa.digest(sa.ROOT/r['path'])==r['sha256']
    frozen=json.loads((FROZEN/'candidate.json').read_text())
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in frozen['manifest'])
    table={(r['dataset'],r['group']):r for r in obj['table']}
    rows=list(csv.DictReader((candidate/'cells.csv').open()));assert len(rows)==2920
    recomputed={}
    for (ds,g),r in table.items():
        rr=[v for v in rows if v['dataset']==ds and v['group']==g];domains={v['domain'] for v in rr}
        assert len(rr)=={'XJTU':55,'MATR':180,'Tongji':130}[ds]
        mm={m:float(np.mean([np.mean([float(v[m]) for v in rr if v['domain']==d]) for d in domains])) for m in ['mae','rmse','p95_ae']}
        assert all(abs(mm[m]-r[m])<1e-10 for m in mm);recomputed[ds,g]=mm
    edges=[('B1','B'),('B2','B'),('B12','B1'),('B12','B2'),('B3','B'),
           ('B13','B1'),('B13','B3'),('B23','B2'),('B23','B3'),
           ('B123','B12'),('B123','B13'),('B123','B23')]
    gates={a+'<'+b:all(recomputed[ds,a][m]<recomputed[ds,b][m]
        for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae']) for a,b in edges}
    assert all(gates.values())
    assert len(obj['paired_comparisons'])==63
    assert objects['stability']['pass_counts']==objects['prediction_audit']['stability_pass_counts']
    limits=['Development datasets reused in model search; global step0.5 chosen after sensitivity, not independent test.',
        'XJTU/MATR incremental MAE intervals cross zero; significance on every dataset not achieved.',
        'Risk sensitivity is fixed-GP risk reweighting, not full retraining; joint7 gates9/10, seed103 retained.',
        'Same-config paired control wins all9 metrics, but independently tuned absolute PCA wins MATR.',
        'Engineering reference-paired supplement, not a claim to invent PCA, GPR, curve differencing or guarantee journal acceptance.',
        'CALCE excluded from M3 selection; this release does not claim independent external M3 confirmation.']
    report=dict(status='VERIFIED_EXPERIMENTAL_RELEASE',candidate=str(candidate.relative_to(sa.ROOT)),
        candidate_sha256=sa.digest(paths['candidate']),variant='wave_pair_pca8_global_step_0.5',
        requirements=dict(full_eight_group_ablation=True,all12_single_module_addition_edges=gates,
            original_models_unchanged=42,new_adapters_hash_verified=84,training_folds_verified=21,
            prediction_rows_independently_audited=29200,paired_comparisons_reported=63,
            risk_pass_counts=objects['stability']['pass_counts'],controls_reported=True,
            source_budget_and_query_boundaries_verified=True),
        evidence={k:dict(path=str(p.relative_to(sa.ROOT)),sha256=sa.digest(p)) for k,p in paths.items()},
        limits=limits,code_sha256=sa.digest(Path(__file__)))
    sa.write_json(out/'acceptance.json',report)
    print(json.dumps(dict(status=report['status'],gates=gates,candidate_sha256=report['candidate_sha256'],acceptance_sha256=sa.digest(out/'acceptance.json'))),flush=True)


if __name__=='__main__':main()
