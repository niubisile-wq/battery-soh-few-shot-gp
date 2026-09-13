"""Diagnostic only: residual head evaluated on one-domain GP ensemble."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import nested_signal_residual as sr


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=sr.ns.rd.sa;root=sa.ROOT/'模块研发/results'
    headroot=root/'m2_nested_signal_residual_v1'
    assert json.loads((headroot/'audit.json').read_text())['status']=='PASS'
    manifest=json.loads((headroot/'result.json').read_text())['manifest'];cells=sa.load_cells('MATR');rows=[]
    for fold in sorted({r['fold'] for r in manifest}):
        train,_,_=sa.split(cells,fold);byid={c.id:c for c in train};allowed=sa.source_indices(train)
        folder=root/'m2_nested_signal_probe_v1/folds'/fold.replace(':','__')
        original=joblib.load(folder/'source_lodo_episodes.joblib');source=json.loads((folder/'result.json').read_text());pair={}
        for excluded in sorted({tuple(f['excluded']) for f in source['fits']}):
            f=next(f for f in source['fits'] if tuple(f['excluded'])==excluded)
            pair[excluded]=joblib.load((root/'m2_nested_signal_probe_v1'/f['artifact']).parent/'episodes.joblib')
        for record in [r for r in manifest if r['fold']==fold]:
            path=headroot/record['artifact'];assert sa.digest(path)==record['sha256'];head=joblib.load(path)
            held=record['held'];parent=record['parent']
            for e in original[parent]:
                if e['domain']!=held:continue
                c=byid[e['cell_id']];q=allowed[c.id][allowed[c.id]>=sa.K]
                delta=sr.ns.signal_views(sa.source_view(c,allowed[c.id]),q)['delta']
                predictions=[]
                for excluded,groups in pair.items():
                    if held not in excluded:continue
                    p=next(p for p in groups[parent] if p['cell_id']==c.id)
                    np.testing.assert_array_equal(p['y'],e['y']);predictions.append(p['base'])
                assert len(predictions)==2
                ensemble=np.mean(predictions,axis=0)
                methods={record['method']:ensemble+sr.correction(delta,head)}
                if record['method']=='rbf_residual':methods['unchanged_ensemble']=ensemble
                for method,p in methods.items():
                    m=sa.metrics(e['y'],p)
                    rows.append(dict(fold=fold,domain=held,cell_id=c.id,parent=parent,method=method,**{k:float(m[k]*100) for k in ['mae','rmse','p95_ae']}))
    summary=[]
    for parent in ['Base','Base+M1']:
        for method in ['unchanged_ensemble','rbf_residual','anchored_residual']:
            groups=defaultdict(list)
            for r in rows:
                if (r['parent'],r['method'])==(parent,method):groups[r['fold'],r['domain']].append(r)
            summary.append(dict(parent=parent,method=method,**{k:float(np.mean([np.mean([r[k] for r in g]) for g in groups.values()])) for k in ['mae','rmse','p95_ae']}))
    sa.write_json(out/'result.json',dict(status='COMPLETE',summary=summary,rows=rows,
        head_result_sha256=sa.digest(headroot/'result.json'),
        limits=['Diagnostic parent predictions changed to average of two separately fitted one-source-domain GPs.',
                'Both constituent GPs exclude scored domain; residual head unchanged and excludes scored domain.',
                'This changes ensembling and domain composition as well as training size, not a pure causal size control.',
                'Not a replacement of frozen GPR/M1 controls, not outer performance, not full M2.']))
    shutil.copyfile(Path(__file__),out/Path(__file__).name);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
