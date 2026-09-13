"""Score frozen source-domain choices and audited original-selection controls."""
import json
from pathlib import Path
from dataclasses import replace
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN
from freeze_source_domain_choices import build_manifest
from support_cv_mixed import SupportCVMixed
from score import PARENTS,addition_edges
from evaluate import error_metrics,aggregate,dump_csv


FAMILIES=('cv','uniform','original_cv','original_uniform')


def group_name(g,family):
    return g+('_'+family if '4' in g and family!='cv' else '')


def main():
    results=HERE.parent/'results';root=results/'m4_source_domain_screen_v1'
    frozen=json.loads((root/'frozen_selections.json').read_text())
    assert frozen==build_manifest(results)
    assert not (root/'score_request.json').exists(),'Existing scoring attempt must be inspected,not overwritten'
    parent=results/'m3_waveform_candidate_v1';old=results/'m4_support_cv_screen_v1'
    sa.write_json(root/'score_request.json',dict(frozen_sha256=sa.digest(root/'frozen_selections.json'),
        code_sha256=sa.digest(Path(__file__)),families=FAMILIES,
        helper_hashes={n:sa.digest(HERE/n) for n in ('support_cv_mixed.py','conditional_mixed.py','freeze_source_domain_choices.py')}))
    cells=[c for ds in ('XJTU','MATR','Tongji') for c in sa.load_cells(ds)]
    rows=[];weights=[];boundary=0.;control_error=0.
    with threadpool_limits(limits=1):
        for fold in frozen['selections']:
            adapters={f:SupportCVMixed(joblib.load(fold['choices'][f]['model']['path']),f) for f in ('cv','uniform')}
            _,_,test=sa.split(cells,fold['fold'])
            for c in test:
                tail=Path('folds')/fold['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(parent/tail) as z:y=z['y'].copy();pp={g:z[g].copy() for g in PARENTS}
                assert np.array_equal(y,c.y[sa.K:])
                for family,model in adapters.items():
                    q=model.predict(sa.inference_view(c));yy=c.y.copy();yy[sa.K:]=999;n=min(sa.K+3,len(yy))
                    boundary=max(boundary,float(np.max(abs(q-model.predict(replace(c,y=yy))))),
                        float(np.max(abs(q[:n-sa.K]-model.predict(sa.prefix(c,n))))))
                    a=fold['choices'][family]['selected']['gain']
                    for g in PARENTS:pp[group_name(g+'4',family)]=pp[g]+a*(q-pp[g])
                    pp['branch_'+family]=q
                    weights.append(dict(fold=fold['fold'],cell_id=c.id,family=family,gain=a,**model.weights(sa.inference_view(c))))
                with np.load(old/tail) as z:
                    assert np.array_equal(y,z['y'])
                    for g in PARENTS:np.testing.assert_array_equal(pp[g],z[g])
                    for family in ('cv','uniform'):
                        f='original_'+family;q=z['branch_'+family]
                        a=fold['choices'][f]['selected']['gain']
                        for g in PARENTS:
                            p=pp[g]+a*(q-pp[g]);oldkey=g+'4'+('_uniform' if family=='uniform' else '')
                            control_error=max(control_error,float(np.max(abs(p-z[oldkey]))));pp[group_name(g+'4',f)]=p
                        pp['branch_'+f]=q.copy()
                path=root/tail;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,y=y,**pp)
                for g,p in pp.items():
                    if not g.startswith('branch_'):rows.append(dict(dataset=c.dataset,domain=c.domain,cell_id=c.id,group=g,**error_metrics(y,p)))
            print(fold['fold'],'source/old-selection controls scored',flush=True)
    assert len(rows)==14600 and len(weights)==730 and max(boundary,control_error)<1e-8
    table=aggregate(rows,['dataset','group']);lookup={(r['dataset'],r['group']):r for r in table}
    failures={f:{a+'<'+b:[dict(dataset=ds,metric=k,delta_pp=lookup[ds,group_name(a,f)][k]-lookup[ds,group_name(b,f)][k])
        for ds in ('XJTU','MATR','Tongji') for k in ('mae','rmse','p95_ae') if lookup[ds,group_name(a,f)][k]>=lookup[ds,group_name(b,f)][k]]
        for a,b in addition_edges()} for f in FAMILIES}
    gates={f:{k:not v for k,v in vv.items()} for f,vv in failures.items()}
    original=json.loads((parent/'candidate.json').read_text())
    parent_error=max(abs(lookup[r['dataset'],r['group']][k]-r[k]) for r in original['table'] for k in ('mae','rmse','p95_ae'))
    assert parent_error<1e-8
    for folder in (parent,FROZEN):
        manifest=json.loads((folder/'candidate.json').read_text());assert all(sa.digest(folder/r['artifact'])==r['sha256'] for r in manifest['manifest'])
    dump_csv(root/'cells.csv',rows);dump_csv(root/'comparison.csv',table);dump_csv(root/'weights.csv',weights)
    sa.write_json(root/'summary.json',dict(status='SCREEN_COMPLETE_NOT_ADOPTED',rows=14600,table=table,gates=gates,failures=failures,
        boundary=boundary,control_replay_error=control_error,parent_metric_error=parent_error,
        limits='Source and original selection controls all reported. No GP refitting in this screen; optimizer warnings remain in frozen choices. Repeated outer development cohorts,independent score replay/uncertainty/stability pending.'))
    print({f:[k for k,v in gg.items() if not v] for f,gg in gates.items()},flush=True)


if __name__=='__main__':main()
