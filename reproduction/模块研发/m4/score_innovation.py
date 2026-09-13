"""Freeze all21 additive/control choices; score32 actual prediction groups."""
import json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN
from conditional_mixed import ConditionalMixed
from score import PARENTS,addition_edges
from evaluate import aggregate,error_metrics,dump_csv


def group_name(g,family):
    return g+('_'+family if family!='cv_innovation' and '4' in g else '')


def main():
    variant='innovation'
    results=HERE.parent/'results';validation=results/('m4_'+variant+'_validation_v1')
    req=json.loads((validation/'request.json').read_text());r=json.loads((validation/'result.json').read_text())
    audit=json.loads((validation/'verification.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(validation/'result.json')
    assert audit['request_sha256']==sa.digest(validation/'request.json')
    assert audit['code_sha256']==sa.digest(HERE/('audit_'+variant+'.py'))
    assert all(sa.digest(HERE/n)==h for n,h in audit['helper_hashes'].items())
    assert all(sa.digest(HERE/n)==h for n,h in req['code_hashes'].items())
    assert len(r['selections'])==len({s['fold'] for s in r['selections']})==21
    families=req['protocol']['families'];selections=[]
    for s in r['selections']:
        root=Path(s['root']);assert sa.digest(root/'result.json')==s['result_sha256']
        fold=json.loads((root/'result.json').read_text());choices={}
        for family in families:
            choice=fold['selected'][family]
            assert choice==min([t for t in fold['trials'] if t['target']==family],key=lambda t:(t['validation_mae'],t['gain'],t['key']))
            kernel=choice['key'].split('__')[1];spec=next(m for m in fold['models'] if m['key']=='mixed__'+kernel)
            assert sa.digest(Path(spec['path']))==spec['sha256'];choices[family]=dict(selected=choice,model=spec)
        selections.append(dict(fold=s['fold'],choices=choices,validation_root=str(root),validation_result_sha256=s['result_sha256']))
    candidate=results/'m3_waveform_candidate_v1';parent=json.loads((candidate/'candidate.json').read_text())
    old=json.loads((FROZEN/'candidate.json').read_text())
    out=results/('m4_'+variant+'_screen_v1');out.mkdir(exist_ok=False)
    sa.write_json(out/'frozen_selections.json',dict(selections=selections,protocol=req['protocol'],
        parent_manifest_sha256=sa.digest(candidate/'candidate.json'),m2_manifest_sha256=sa.digest(FROZEN/'candidate.json'),
        validation_audit_sha256=sa.digest(validation/'verification.json'),code_sha256=sa.digest(Path(__file__))))
    cells=[c for ds in ('XJTU','MATR','Tongji') for c in sa.load_cells(ds)];rows=[];weight_rows=[];boundary=0.;bias_error=0.
    with threadpool_limits(limits=1):
        for s in selections:
            adapters={}
            for family,choice in s['choices'].items():
                model=joblib.load(choice['model']['path'])
                from support_cv_mixed import SupportCVMixed
                from innovation_adapter import InnovationAdapter
                adapters[family]=SupportCVMixed(model,'cv') if family=='cv_blend' else InnovationAdapter(model,'uniform' if family=='uniform_innovation' else 'cv')
            _,_,test=sa.split(cells,s['fold'])
            for c in test:
                tail=Path('folds')/s['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(candidate/tail) as z:y=z['y'];pp={g:z[g] for g in PARENTS}
                assert np.array_equal(y,c.y[sa.K:])
                for family,adapter in adapters.items():
                    q=adapter.predict(sa.inference_view(c));yy=c.y.copy();yy[sa.K:]=999;n=min(len(c.y),sa.K+3)
                    boundary=max(boundary,float(np.max(abs(q-adapter.predict(replace(c,y=yy))))),
                        float(np.max(abs(q[:n-sa.K]-adapter.predict(sa.prefix(c,n))))))
                    gain=s['choices'][family]['selected']['gain']
                    w=(adapter if family=='cv_blend' else adapter.adapter).weights(sa.inference_view(c))
                    weight_rows.append(dict(fold=s['fold'],cell_id=c.id,family=family,gain=gain,**w))
                    for g in PARENTS:pp[group_name(g+'4',family)]=pp[g]+gain*(q-pp[g] if family=='cv_blend' else q)
                    pp['branch_'+family]=q
                with np.load(results/'m4_support_cv_screen_v1'/tail) as oldpred:
                    for g in PARENTS:bias_error=max(bias_error,float(np.max(abs(pp[g+'4_cv_blend']-oldpred[g+'4']))))
                path=out/tail;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,**pp)
                for g,p in pp.items():
                    if not g.startswith('branch_'):rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,p)))
            print(s['fold'],'all families scored',flush=True)
    assert len(rows)==11680 and max(boundary,bias_error)<1e-8
    table=aggregate(rows,['dataset','group']);look={(t['dataset'],t['group']):t for t in table}
    failures={family:{a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=look[ds,group_name(a,family)][k]-look[ds,group_name(b,family)][k])
        for ds in ('XJTU','MATR','Tongji') for k in ('mae','rmse','p95_ae')
        if look[ds,group_name(a,family)][k]>=look[ds,group_name(b,family)][k]] for a,b in addition_edges()} for family in families}
    gates={f:{k:not v for k,v in rr.items()} for f,rr in failures.items()}
    parent_error=max(abs(look[t['dataset'],t['group']][k]-t[k]) for t in parent['table'] for k in ('mae','rmse','p95_ae'))
    assert parent_error<1e-8
    assert all(sa.digest(candidate/t['artifact'])==t['sha256'] for t in parent['manifest'])
    assert all(sa.digest(FROZEN/t['artifact'])==t['sha256'] for t in old['manifest'])
    dump_csv(out/'cells.csv',rows);dump_csv(out/'comparison.csv',table)
    assert len(weight_rows)==1095
    dump_csv(out/'weights.csv',weight_rows)
    sa.write_json(out/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',rows=len(rows),table=table,failures=failures,gates=gates,
        boundary=boundary,blend_replay_error=bias_error,parent_metric_error=parent_error,
        limits='Branch innovation arrays are corrections,not standalone SOH predictions; excluded from metrics. Repeated development cohorts. Independent scoring audit,paired intervals and source-risk stability pending.'))
    print({f:[k for k,v in g.items() if not v] for f,g in gates.items()},flush=True)


if __name__=='__main__':main()


