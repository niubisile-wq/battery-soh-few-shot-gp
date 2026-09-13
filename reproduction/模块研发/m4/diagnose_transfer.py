"""Descriptive residual-direction diagnosis; never produces inference gates."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa, HERE
from evaluate import dump_csv


def main():
    results = HERE.parent / 'results'; out = results / 'm4_transfer_diagnosis_v1'
    out.mkdir(exist_ok=False)
    sources = json.loads((results / 'm4_source_batch_v1/result.json').read_text())['folds']
    source_by_fold = {r['fold']: Path(r['source']) for r in sources}
    plain_by_fold = {r['fold']: Path(r['correction']) for r in sources}
    datasets = {ds: {c.id: c for c in sa.load_cells(ds)} for ds in ('XJTU', 'MATR', 'Tongji')}
    source_stats = {}; validation_stats = {}
    for fold, path in source_by_fold.items():
        episodes = joblib.load(path / 'complete_parent_source_oof.joblib')
        by_domain = defaultdict(list)
        for e in episodes:
            error = (e['predictions']['B123'] - e['y']) * 100
            by_domain[e['domain']].append(float(np.mean(error)))
        source_stats[fold] = {d: float(np.mean(v)) for d, v in by_domain.items()}
        # These predictions are shared frozen parents, not fourth-module fits.
        vp = joblib.load(plain_by_fold[fold] / 'validation_parents.joblib')
        by_domain = defaultdict(list)
        for (cid, g), pred in vp.items():
            if g != 'B123': continue
            c = datasets[fold.split(':')[0]][cid]
            by_domain[c.domain].append(float(np.mean((pred - c.y[sa.K:]) * 100)))
        validation_stats[fold] = {d: float(np.mean(v)) for d, v in by_domain.items()}
    rows = []; hashes = {}
    for family, directory in [('plain', 'm4_residual_screen_v1'), ('anchored', 'm4_anchored_screen_v1'),
                              ('condition', 'm4_condition_screen_v1'), ('consensus', 'm4_consensus_screen_v1'),
                              ('robust', 'm4_robust_screen_v1')]:
        root = results / directory
        hashes[family] = sa.digest(root / 'summary.json')
        frozen = json.loads((root / 'frozen_selections.json').read_text())['selections']
        for selection in frozen:
            fold = selection['fold']; ds, domain = fold.split(':')
            for c in datasets[ds].values():
                if c.domain != domain: continue
                path = root / 'folds' / fold.replace(':', '__') / 'predictions' / (c.id + '.npz')
                with np.load(path) as z:
                    assert np.array_equal(z['y'], c.y[sa.K:])
                    before = (np.asarray(z['B123'], float) - np.asarray(z['y'], float)) * 100
                    after = (np.asarray(z['B1234'], float) - np.asarray(z['y'], float)) * 100
                    change = after - before
                    # Negative before*change points initially toward zero; large steps may overshoot.
                    moving = np.abs(change) > 1e-10
                    rows.append(dict(family=family, dataset=ds, domain=domain, cell_id=c.id,
                        gain=selection['selected']['gain'], key=selection['selected']['key'],
                        parent_bias_pp=float(before.mean()), correction_pp=float(change.mean()),
                        parent_mae_pp=float(np.abs(before).mean()), mae_delta_pp=float((abs(after)-abs(before)).mean()),
                        fraction_changed=float(moving.mean()),
                        fraction_direction_toward_zero=float(np.mean((before*change)[moving] < 0)) if moving.any() else None,
                        fraction_error_improved=float(np.mean(abs(after)<abs(before)-1e-10))))
    aggregated = []
    for family in hashes:
        for ds in datasets:
            domains = sorted({r['domain'] for r in rows if r['family']==family and r['dataset']==ds})
            for domain in domains:
                rr = [r for r in rows if (r['family'],r['dataset'],r['domain'])==(family,ds,domain)]
                fold=ds+':'+domain
                fields = ('parent_bias_pp','correction_pp','parent_mae_pp','mae_delta_pp','fraction_changed','fraction_error_improved')
                aggregated.append(dict(family=family,dataset=ds,domain=domain,cells=len(rr),
                    **{k:float(np.mean([r[k] for r in rr])) for k in fields},
                    source_oof_domain_bias_pp=source_stats[fold], validation_domain_bias_pp=validation_stats[fold]))
    dump_csv(out / 'cells.csv', rows)
    sa.write_json(out / 'summary.json', dict(status='DESCRIPTIVE_DIAGNOSIS_COMPLETE', rows=len(rows),
        domains=aggregated, source_summary_hashes=hashes, code_sha256=sa.digest(Path(__file__)),
        limits=['Query truth used only for retrospective diagnostics, never inference gates or per-domain switches.',
                'Source OOF and validation bias are different populations/models; differences alone do not establish causality.',
                'No new candidate selected, no refitting, no independent confirmation.']))
    for r in aggregated:
        if r['dataset']=='XJTU' and r['domain']=='Sim_satellite': print(json.dumps(r),flush=True)


if __name__=='__main__': main()
