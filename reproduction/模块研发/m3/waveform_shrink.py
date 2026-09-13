"""Uniform-step sensitivity: never mix best steps across datasets or folds."""
import json
import argparse
from pathlib import Path
import numpy as np
from screen import sa,HERE,GROUPS,error_metrics,aggregate,dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    p=json.loads((HERE/'waveform_shrink_protocol.json').read_text());root=HERE.parent/'results'/p['source_run']
    sa.write_json(out/'protocol.json',dict(protocol=p,code_sha256=sa.digest(Path(__file__))))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[]
    for c in cells:
        path=root/'folds'/(c.dataset+'__'+c.domain)/'predictions'/(c.id+'.npz')
        with np.load(path) as z:
            assert np.array_equal(z['y'],c.y[sa.K:])
            for g in GROUPS:rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(z['y'],z[g])))
            for family in p['families']:
                for step in p['steps']:
                    tag=family+'__'+str(step)
                    for g in GROUPS:
                        pred=z[g]+step*(z[g+'3__'+family]-z[g])
                        rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g+'3__'+tag,**error_metrics(z['y'],pred)))
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={}
    edges=[('B3','B'),('B123','B12'),('B123','B13'),('B123','B23'),('B123','B3'),('B13','B1'),('B23','B2')]
    for family in p['families']:
        for step in p['steps']:
            tag=family+'__'+str(step)
            gates[tag]={a+'<'+b:all(look[ds,a+'__'+tag][m]<look[ds,b if b in GROUPS else b+'__'+tag][m]
                for ds in ['XJTU','MATR','Tongji'] for m in ['mae','rmse','p95_ae']) for a,b in edges}
    assert len(rows)==365*68
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    passing=[k for k,g in gates.items() if all(g.values())]
    sa.write_json(out/'summary.json',dict(status='SENSITIVITY_COMPLETE_NOT_ADOPTED',table=table,gates=gates,passing_point_candidates=passing,cells=365,folds=21))
    print(json.dumps(dict(passing_point_candidates=passing,gates=gates)),flush=True)


if __name__=='__main__':main()
