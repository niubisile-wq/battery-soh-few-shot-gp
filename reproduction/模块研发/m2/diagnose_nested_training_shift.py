"""Matched source-query comparison of one-domain and two-domain GP fits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import nested_signal_probe as ns


def compare(y,one,two):
    a=y-one;b=y-two
    return dict(one_mae_pp=float(abs(a).mean()*100),two_mae_pp=float(abs(b).mean()*100),
                residual_shift_mae_pp=float(abs(a-b).mean()*100),
                one_residual_bias_pp=float(a.mean()*100),two_residual_bias_pp=float(b.mean()*100),
                opposite_sign_fraction=float(np.mean(a*b<0)),
                shift_exceeds_two_error_fraction=float(np.mean(abs(a-b)>abs(b))))


def aggregate(rows):
    result=[]
    for parent in sorted({r['parent'] for r in rows}):
        cells=defaultdict(list)
        for r in rows:
            if r['parent']==parent:cells[r['fold'],r['domain'],r['cell_id']].append(r)
        domains=defaultdict(list)
        keys=list(compare(np.ones(2),np.zeros(2),np.zeros(2)))
        for (fold,domain,cell_id),rr in cells.items():
            domains[fold,domain].append({k:float(np.mean([r[k] for r in rr])) for k in keys})
        result.append(dict(parent=parent,cell_pairs=sum(map(len,cells.values())),**{k:float(np.mean([np.mean([r[k] for r in rr]) for rr in domains.values()])) for k in keys}))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=ns.rd.sa
    root=sa.ROOT/'模块研发/results/m2_nested_signal_probe_v1'
    assert json.loads((root/'audit.json').read_text())['status']=='PASS'
    protocol=json.loads((root/'protocol.json').read_text());allcells=sa.load_cells('MATR');rows=[];inputs=[]
    for fold in protocol['folds']:
        train,_,_=sa.split(allcells,fold);byid={c.id:c for c in train};allowed=sa.source_indices(train)
        folder=root/'folds'/fold.replace(':','__');result=json.loads((folder/'result.json').read_text())
        path=folder/'source_lodo_episodes.joblib';original=joblib.load(path);inputs.append(dict(path=str(path),sha256=sa.digest(path)))
        for excluded in sorted({tuple(f['excluded']) for f in result['fits']}):
            fit=next(f for f in result['fits'] if tuple(f['excluded'])==excluded)
            path=(root/fit['artifact']).parent/'episodes.joblib';pair=joblib.load(path);inputs.append(dict(path=str(path),sha256=sa.digest(path)))
            for parent,field in [('Base','base'),('Base+M1','base'),('Physical_control','physical')]:
                key=parent if parent!='Physical_control' else 'Base';reference={e['cell_id']:e for e in original[key]}
                for e in pair[key]:
                    c=byid[e['cell_id']];q=allowed[c.id][allowed[c.id]>=sa.K];r=reference[c.id]
                    assert c.domain in excluded and e['domain']==r['domain']==c.domain
                    assert e['query_indices']==q.tolist()
                    np.testing.assert_array_equal(e['y'],r['y']);np.testing.assert_array_equal(e['y'],c.y[q])
                    rows.append(dict(fold=fold,domain=c.domain,cell_id=c.id,parent=parent,excluded=list(excluded),
                                     **compare(e['y'],e[field],r[field])))
    summary=aggregate(rows)
    sa.write_json(out/'result.json',dict(status='COMPLETE',summary=summary,rows=rows,inputs=inputs,
        limits=['Matched original-budget source queries; no outer target scoring or new labels.',
                'Adding a domain changes sample count, domain composition, feature fitting and GP kernel optimization together; not a pure sample-size causal experiment.',
                'Residual shift does not prove it caused prior residual-head failures.',
                'Do not reuse a two-domain residual as meta-training data when that GP fitted the meta-held domain.']))
    shutil.copyfile(Path(__file__),out/Path(__file__).name);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
