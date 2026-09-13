"""Frozen support gate:16groups,8matched ungated controls and direct branch."""
from dataclasses import replace
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN
from conditional_geometry import ConditionalGeometry
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registered',action='store_true');args=ap.parse_args()
    prefix='m4_registered_gate' if args.registered else 'm4_support_gate'
    results=HERE.parent/'results';validation=results/(prefix+'_validation_v1')
    audit=json.loads((validation/'verification.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(validation/'result.json')
    frozen=json.loads((validation/'result.json').read_text());selections=frozen['selections'];assert len(selections)==21
    parent=results/'m3_waveform_candidate_v1';manifest=json.loads((parent/'candidate.json').read_text())
    out=results/(prefix+'_screen_v1');out.mkdir(exist_ok=False)
    for s in selections:
        assert sa.digest(Path(s['result']))==s['result_sha256']
        assert sa.digest(Path(s['gate']['path']))==s['gate']['sha256']
    sa.write_json(out/'frozen_selections.json',dict(selections=selections,validation_sha256=sa.digest(validation/'result.json'),
        parent_sha256=sa.digest(parent/'candidate.json'),code_sha256=sa.digest(Path(__file__))))
    cells=[c for ds in ('XJTU','MATR','Tongji') for c in sa.load_cells(ds)];rows=[];diagnostics=[];boundary=0.
    with threadpool_limits(limits=1):
        for s in selections:
            gate=joblib.load(s['gate']['path']);choice=s['selected'];mode=choice['key'].split('__')[-1]
            branch=ConditionalGeometry(gate.model,mode);_,_,test=sa.split(cells,s['fold'])
            for c in test:
                tail=Path('folds')/s['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(parent/tail) as z:y=z['y'];pp={g:z[g] for g in PARENTS}
                assert np.array_equal(y,c.y[sa.K:])
                q=branch.predict(sa.inference_view(c));weight=gate.predict(c,choice['multiplier']);gain=choice['gain']
                yy=c.y.copy();yy[sa.K:]=999;n=min(len(c.x),sa.K+3);short=sa.prefix(c,n)
                boundary=max(boundary,float(np.max(abs(q-branch.predict(replace(c,y=yy))))),
                    float(np.max(abs(weight-gate.predict(replace(c,y=yy),choice['multiplier'])))),
                    float(np.max(abs(q[:n-sa.K]-branch.predict(short)))),
                    float(np.max(abs(weight[:n-sa.K]-gate.predict(short,choice['multiplier'])))))
                for g in PARENTS:
                    pp[g+'4']=pp[g]+gain*weight*(q-pp[g])
                    pp[g+'4_ungated']=pp[g]+gain*(q-pp[g])
                pp['branch_only']=q
                path=out/tail;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,gate=weight,**pp)
                for g,pred in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,pred)))
                diagnostics.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,gain=gain,mean_gate=float(weight.mean())))
            print(s['fold'],'gated16groups and matched controls scored',flush=True)
    assert len(rows)==9125 and boundary<1e-8
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    failures={a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=look[ds,a][k]-look[ds,b][k]) for ds in ('XJTU','MATR','Tongji') for k in ('mae','rmse','p95_ae') if look[ds,a][k]>=look[ds,b][k]] for a,b in addition_edges()}
    gates={k:not v for k,v in failures.items()}
    assert all(sa.digest(parent/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    original=json.loads((FROZEN/'candidate.json').read_text());assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in original['manifest'])
    parent_error=max(abs(look[r['dataset'],r['group']][k]-r[k]) for r in manifest['table'] for k in ('mae','rmse','p95_ae'))
    assert parent_error<1e-8
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table);dump_csv(out/'gate_diagnostics.csv',diagnostics)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,failures=failures,
        all32_edges_pass=all(gates.values()),rows=len(rows),max_boundary_error=boundary,parent_metric_error=parent_error,
        limits='Reused development cohorts,not independent confirmation. Outer score audit and paired/risk stability pending.'))
    print(json.dumps(dict(all32_edges_pass=all(gates.values()),failed=[k for k,v in gates.items() if not v])),flush=True)


if __name__=='__main__':main()
