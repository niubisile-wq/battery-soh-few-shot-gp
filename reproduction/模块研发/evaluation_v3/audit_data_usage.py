"""Historical scored-cell exposure and preprocessing-quality metadata audit."""
import argparse
from collections import defaultdict, Counter
import csv
import json
from pathlib import Path
from evaluate import sa, dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    results=sa.ROOT/'开发基线选择依据/results'
    history=['multidataset_hi_matrix/matrix_cell_results.csv',
             'self_dataset_hi/self_cell_results.csv','matr_cross_hi/matr_cell_results.csv',
             'core_p4_p5/p4_p5_cell_results.csv','legacy_rpt_external/legacy_results.csv']
    exposure=[]
    for name in history:
        path=results/name
        with path.open() as f:rows=list(csv.DictReader(f))
        group=defaultdict(set); counts=Counter()
        for r in rows:
            ds=r.get('target',r.get('dataset'))
            group[ds].add(r['cell_id']);counts[ds]+=1
        for ds,ids in sorted(group.items()):
            exposure.append(dict(dataset=ds,artifact=str(path.relative_to(sa.ROOT)),
                sha256=sa.digest(path),score_rows=counts[ds],unique_cells=len(ids),cell_ids=sorted(ids),
                conclusion='Historical scoring exposure confirmed; not evidence that v3 was tuned on these data'))
    untouched={ds:False for ds in ['HUST','SNL','CALCE','Oxford','NASA']}
    assert all(any(r['dataset']==ds for r in exposure) for ds in untouched)
    manifest_paths={'XJTU':sa.ROOT/'实验部署/source_window_cache_v2/manifest.json',
                    **{ds:sa.ROOT/f'实验部署/m1_development_cache_v1/{ds}/manifest.json' for ds in ['MATR','Tongji']}}
    with (sa.ROOT/'模块研发/results/frozen_v3_evaluation_v2/cells.csv').open() as f:
        errors={r['cell_id']:r for r in csv.DictReader(f) if r['group']=='Full_v3'}
    quality=[];inputs=[]
    signal_reasons={'Time must be strictly increasing within cycle','No complete contiguous charge window',
                    'Missing or nonfinite signal','Signal shape mismatch'}
    for ds,path in manifest_paths.items():
        d=json.loads(path.read_text());inputs.append(dict(path=str(path.relative_to(sa.ROOT)),sha256=sa.digest(path)))
        for c in d['cells']:
            if ds=='XJTU' and c['candidate']!='offset_03_01':continue
            if c['cell_id'] not in errors:continue
            reasons=(Counter({r['reason']:r['count'] for r in d['rejection_counts']
                             if r['cell_id']==c['cell_id'] and r['candidate']=='offset_03_01'})
                     if ds=='XJTU' else Counter(c['rejections']))
            assert sum(reasons.values()) == c['total_cycles']-c['valid_cycles']
            bad=sum(n for reason,n in reasons.items() if reason in signal_reasons)
            other=sum(reasons.values())-bad
            row=errors[c['cell_id']]
            quality.append(dict(dataset=ds,domain=row['domain'],cell_id=c['cell_id'],total_cycles=c['total_cycles'],
                valid_cycles=c['valid_cycles'],signal_rejected_cycles=bad,other_rejected_cycles=other,
                signal_reject_fraction=bad/c['total_cycles'],full_v3_mae_pp=float(row['mae'])))
    assert len(quality)==365
    summary=[]
    for ds in manifest_paths:
        rr=[r for r in quality if r['dataset']==ds]
        summary.append(dict(dataset=ds,cells=len(rr),total_cycles=sum(r['total_cycles'] for r in rr),
            valid_cycles=sum(r['valid_cycles'] for r in rr),signal_rejected_cycles=sum(r['signal_rejected_cycles'] for r in rr),
            other_rejected_cycles=sum(r['other_rejected_cycles'] for r in rr),
            cells_with_signal_rejection=sum(r['signal_rejected_cycles']>0 for r in rr)))
    report=dict(status='COMPLETE',code_sha256=sa.digest(Path(__file__)),historical_exposure=exposure,
        untouched_claim_supported=untouched,quality_summary=summary,quality_manifest_inputs=inputs,
        limits=['Prior scoring is not proof of direct v3 tuning, but disallows claiming all eight datasets were untouched.',
                'Coverage exclusions are not all signal faults: aging-vs-RPT and discharge capacity exclusions are separate.',
                'First failure is recorded by preprocessing; later signal faults can be masked by earlier capacity rejection.',
                'No predictions exist for rejected windows; their SOH accuracy is unknown, not zero error.',
                'Raw observation density within accepted windows is not preserved in the current cache.',
                'No external target labels opened; historical score files and metadata only.',
                'A new untouched data source or verified unused cohort is required for strong independent final-test claim.'])
    sa.write_json(out/'audit.json',report);dump_csv(out/'quality_by_cell.csv',quality)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
