"""Freeze both target families across all21 folds before any query scoring."""
from dataclasses import replace
import json,argparse
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def name(group,target):
    return group+('_'+target if target in ('direct','zero','cell_only') and '4' in group else '')


def main():
    ap=argparse.ArgumentParser();variants=ap.add_mutually_exclusive_group();variants.add_argument('--mixed-effects',action='store_true');variants.add_argument('--hierarchical',action='store_true');args=ap.parse_args()
    variant='hierarchical' if args.hierarchical else 'mixed_effects' if args.mixed_effects else 'difference'
    families=('hierarchical','cell_only') if args.hierarchical else ('mixed','zero') if args.mixed_effects else ('difference','direct')
    results=HERE.parent/'results';batchroot=results/('m4_'+variant+'_batch_v1')
    batch=json.loads((batchroot/'result.json').read_text())
    assert batch['status']=='VALIDATION_BATCH_COMPLETE' and len({r['fold'] for r in batch['folds']})==21
    candidate=results/'m3_waveform_candidate_v1';parent=json.loads((candidate/'candidate.json').read_text())
    old=json.loads((FROZEN/'candidate.json').read_text());selections=[]
    for e in batch['folds']:
        root=Path(e['branch']);req=json.loads((root/'request.json').read_text());r=json.loads((root/'result.json').read_text())
        audit=json.loads((root/'verification.json').read_text())
        assert req['fold']==e['fold'] and e['status']=='VALIDATION_AUDITED'
        assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
        assert audit['request_sha256']==sa.digest(root/'request.json') and audit['code_sha256']==sa.digest(HERE/('audit_'+variant+'.py'))
        if args.hierarchical:assert audit['helper_sha256']==sa.digest(HERE/'audit_conditional_mixed.py')
        assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
        choices={}
        for target in families:
            choice=r['selected'][target]
            assert choice==min([t for t in r['trials'] if t['target']==target],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
            spec=next(m for m in r['models'] if m['key']==choice['key'])
            assert sa.digest(Path(spec['path']))==spec['sha256']
            choices[target]=dict(selected=choice,model=spec)
        selections.append(dict(fold=e['fold'],choices=choices,branch_root=str(root),result_sha256=sa.digest(root/'result.json')))
    out=results/('m4_'+variant+'_screen_v1');out.mkdir(exist_ok=False)
    sa.write_json(out/'frozen_selections.json',dict(selections=selections,parent_manifest_sha256=sa.digest(candidate/'candidate.json'),
        m2_manifest_sha256=sa.digest(FROZEN/'candidate.json'),code_sha256=sa.digest(Path(__file__)),
        protocol=json.loads((HERE/(variant+'_protocol.json')).read_text()),limits='Repeated development data, not independent confirmation.'))
    cells=[c for ds in ('XJTU','MATR','Tongji') for c in sa.load_cells(ds)]
    rows=[];boundary=0.;control_error=0.
    with threadpool_limits(limits=1):
        for s in selections:
            models={t:joblib.load(v['model']['path']) for t,v in s['choices'].items()}
            _,_,test=sa.split(cells,s['fold'])
            for c in test:
                tail=Path('folds')/s['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(candidate/tail) as z:y=z['y'];pp={g:z[g] for g in PARENTS}
                assert np.array_equal(y,c.y[sa.K:])
                for target,m in models.items():
                    q=m.predict(sa.inference_view(c));yy=c.y.copy();yy[sa.K:]=999;n=min(len(c.x),sa.K+3)
                    boundary=max(boundary,float(np.max(abs(q-m.predict(replace(c,y=yy))))),
                        float(np.max(abs(q[:n-sa.K]-m.predict(sa.prefix(c,n))))))
                    gain=s['choices'][target]['selected']['gain']
                    for g in PARENTS:pp[name(g+'4',target)]=pp[g]+gain*(q-pp[g])
                    pp['branch_'+target]=q
                if args.hierarchical:
                    with np.load(results/'m4_evidence_screen_v1'/tail) as control:
                        for g in PARENTS:control_error=max(control_error,float(np.max(abs(pp[g+'4_cell_only']-control[g+'4_uniform']))))
                path=out/tail;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,**pp)
                for g,p in pp.items():rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,p)))
            print(s['fold'],'both families scored',flush=True)
    assert len(rows)==9490 and max(boundary,control_error)<1e-8
    table=aggregate(rows,['dataset','group']);look={(r['dataset'],r['group']):r for r in table}
    failures={}
    for target in families:
        failures[target]={a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=look[ds,name(a,target)][k]-look[ds,name(b,target)][k])
            for ds in ('XJTU','MATR','Tongji') for k in ('mae','rmse','p95_ae')
            if look[ds,name(a,target)][k]>=look[ds,name(b,target)][k]] for a,b in addition_edges()}
    gates={t:{k:not v for k,v in f.items()} for t,f in failures.items()}
    error=max(abs(look[r['dataset'],r['group']][k]-r[k]) for r in parent['table'] for k in ('mae','rmse','p95_ae'))
    assert error<1e-8
    assert all(sa.digest(candidate/r['artifact'])==r['sha256'] for r in parent['manifest'])
    assert all(sa.digest(FROZEN/r['artifact'])==r['sha256'] for r in old['manifest'])
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',table=table,gates=gates,failures=failures,
        rows=len(rows),max_boundary_error=boundary,parent_metric_error=error,control_replay_error=control_error,
        limits='Both families reported. Independent scoring replay,paired uncertainty and stability pending. Repeated development data.'))
    print({t:[k for k,v in g.items() if not v] for t,g in gates.items()},flush=True)


if __name__=='__main__':main()
