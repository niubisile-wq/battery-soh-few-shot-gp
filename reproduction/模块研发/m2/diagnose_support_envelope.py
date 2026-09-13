"""Source-budget pilot of fixed support-maximum output envelope."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import repair_diagnostics as rd


def envelope(prediction,support):
    # Upper bound only; no assertion that noisy future labels obey this bound.
    return np.minimum(prediction,float(np.max(support)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=rd.sa
    screen=sa.ROOT/'模块研发/results/m2_reference_raw_backbone_screen_v1'
    protocol=json.loads((screen/'protocol.json').read_text());rows=[];inputs=[]
    for dataset in sa.PROTOCOL['datasets']:
        cells=sa.load_cells(dataset)
        for fold in protocol['folds']:
            if not fold.startswith(dataset+':'):continue
            train,_,_=sa.split(cells,fold);byid={c.id:c for c in train};allowed=sa.source_indices(train)
            path=screen/'folds'/fold.replace(':','__')/'seed_0/source_lodo_episodes.joblib'
            episodes=joblib.load(path);inputs.append(dict(path=str(path),sha256=sa.digest(path)))
            for parent,field in [('Base','base'),('Base+M1','base'),('Physical_control','physical')]:
                for e in episodes[parent if parent!='Physical_control' else 'Base']:
                    c=byid[e['cell_id']];q=allowed[c.id][allowed[c.id]>=sa.K];np.testing.assert_array_equal(e['y'],c.y[q])
                    y=e['y'];p=e[field];support=c.y[:sa.K];bounded=envelope(p,support)
                    old=sa.metrics(y,p);new=sa.metrics(y,bounded)
                    rows.append(dict(dataset=dataset,fold=fold,domain=c.domain,cell_id=c.id,parent=parent,
                        future_above_support_fraction=float(np.mean(y>support.max())),changed_fraction=float(np.mean(p!=bounded)),
                        harmed_fraction=float(np.mean(abs(bounded-y)>abs(p-y)+1e-12)),
                        **{k+'_delta_pp':float(100*(new[k]-old[k])) for k in ['mae','rmse','p95_ae']}))
    summary=[];keys=['future_above_support_fraction','changed_fraction','harmed_fraction','mae_delta_pp','rmse_delta_pp','p95_ae_delta_pp']
    for dataset in sa.PROTOCOL['datasets']:
        for parent in ['Base','Base+M1','Physical_control']:
            groups=defaultdict(list)
            for r in rows:
                if (r['dataset'],r['parent'])==(dataset,parent):groups[r['fold'],r['domain']].append(r)
            summary.append(dict(dataset=dataset,parent=parent,**{k:float(np.mean([np.mean([r[k] for r in g]) for g in groups.values()])) for k in keys}))
    sa.write_json(out/'result.json',dict(status='COMPLETE',summary=summary,rows=rows,inputs=inputs,
        limits=['Only original source OOF budget queries; not outer tests or v8 full predictions.',
                'Support maximum is an empirical constraint, not a guaranteed upper bound on noisy future SOH.',
                'No new labels, no learned threshold; future labels used only for retrospective scoring.']))
    shutil.copyfile(Path(__file__),out/Path(__file__).name);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
