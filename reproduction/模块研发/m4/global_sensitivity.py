"""All prescribed global scalar steps across seven declared M4 sources."""
import argparse
import json
from pathlib import Path
import numpy as np
from source_oof import sa,HERE
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=HERE.parent/'results';p=json.loads((HERE/'condition_controls_protocol.json').read_text())
    sources={'plain':('m4_residual_screen_v1',''),'anchored':('m4_anchored_screen_v1',''),
        'multimetric_balanced':('m4_multimetric_screen_v1','__balanced'),'multimetric_constrained':('m4_multimetric_screen_v1','__constrained'),
        'condition_mixed':('m4_condition_screen_v1',''),'condition_geometry_raw':('m4_condition_controls_v1','__geometry_raw'),
        'condition_geometry_residualized':('m4_condition_controls_v1','__geometry_residualized')}
    assert set(sources)==set(p['sensitivity_sources'])
    sa.write_json(out/'protocol.json',dict(protocol=p,sources=sources,code_sha256=sa.digest(Path(__file__)),
        source_summary_hashes={k:sa.digest(root/v[0]/'summary.json') for k,v in sources.items()}))
    cells=[c for ds in ['XJTU','MATR','Tongji'] for c in sa.load_cells(ds)];rows=[]
    for c in cells:
        tail=Path('folds')/(c.dataset+'__'+c.domain)/'predictions'/(c.id+'.npz')
        with np.load(root/'m3_waveform_candidate_v1'/tail) as z:y=z['y'];base={g:z[g] for g in PARENTS}
        assert np.array_equal(y,c.y[sa.K:])
        for g,pred in base.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,pred)))
        for family,(dirname,suffix) in sources.items():
            with np.load(root/dirname/tail) as z:
                assert np.array_equal(z['y'],y)
                for g in PARENTS:
                    assert np.max(abs(z[g]-base[g]))<1e-8
                    delta=z[g+'4'+suffix]-base[g]
                    for step in p['sensitivity_steps']:
                        pred=base[g]+step*delta;tag=family+'__'+str(step)
                        rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g+'4__'+tag,**error_metrics(y,pred)))
    assert len(rows)==105120
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table};gates={};failures={}
    for family in sources:
        for step in p['sensitivity_steps']:
            tag=family+'__'+str(step);group=lambda g:g+'__'+tag if '4' in g else g
            failures[tag]={a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=look[ds,group(a)][k]-look[ds,group(b)][k])
                for ds in ['XJTU','MATR','Tongji'] for k in ['mae','rmse','p95_ae'] if look[ds,group(a)][k]>=look[ds,group(b)][k]] for a,b in addition_edges()}
            gates[tag]={k:not v for k,v in failures[tag].items()}
    passing=[k for k,g in gates.items() if all(g.values())]
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SENSITIVITY_COMPLETE_NOT_ADOPTED',table=table,gates=gates,failures=failures,passing_candidates=passing,
        limits=p['limits'],rows=len(rows)))
    print(json.dumps(dict(passing_candidates=passing)),flush=True)


if __name__=='__main__':main()
