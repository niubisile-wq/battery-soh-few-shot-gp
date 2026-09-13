"""Independent source-selected and original-control predictions and intervals."""
import csv,itertools,json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE,FROZEN
from audit_support_cv import independent_weight
from audit_conditional_mixed import independent_predict
from summarize_strict import paired_stats


def mapped(g,f):
    return g+('_'+f if '4' in g and f!='cv' else '')


def main():
    results=HERE.parent/'results';root=results/'m4_source_domain_screen_v1';parent=results/'m3_waveform_candidate_v1'
    frozen=json.loads((root/'frozen_selections.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert sa.digest(parent/'candidate.json')==frozen['parent_manifest_sha256']
    assert sa.digest(FROZEN/'candidate.json')==frozen['m2_manifest_sha256']
    from freeze_source_domain_choices import build_manifest
    assert frozen==build_manifest(results)
    score_req=json.loads((root/'score_request.json').read_text())
    assert score_req['frozen_sha256']==sa.digest(root/'frozen_selections.json')
    assert score_req['code_sha256']==sa.digest(HERE/'score_source_domains.py')
    assert all(sa.digest(HERE/n)==h for n,h in score_req['helper_hashes'].items())
    control=results/'m4_support_cv_screen_v1'
    parents=['B'+''.join(s) for n in range(4) for s in itertools.combinations('123',n)]
    families=('cv','uniform','original_cv','original_uniform');datasets=('XJTU','MATR','Tongji')
    rows=list(csv.DictReader((root/'cells.csv').open()));lookup={(r['cell_id'],r['group']):r for r in rows}
    weights=list(csv.DictReader((root/'weights.csv').open()));wl={(r['cell_id'],r['family']):r for r in weights}
    assert len(rows)==len(lookup)==14600 and len(weights)==len(wl)==730
    cells=[c for ds in datasets for c in sa.load_cells(ds)]
    errors=dict(prediction=0.,metric=0.,weight=0.,control=0.);buckets=defaultdict(list);cm={};seen=set()
    with threadpool_limits(limits=1):
        for fold in frozen['selections']:
            vr=Path(fold['full_source_record']);assert sa.digest(vr)==fold['full_source_record_sha256']
            models={}
            for f,s in fold['choices'].items():
                assert sa.digest(Path(s['model']['path']))==s['model']['sha256'];models[f]=joblib.load(s['model']['path'])
            _,_,test=sa.split(cells,fold['fold'])
            for c in test:
                tail=Path('folds')/fold['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(parent/tail) as z:y=z['y'].copy();expected={g:z[g].copy() for g in parents}
                assert np.array_equal(y,c.y[sa.K:])
                for f,model in models.items():
                    masked=sa.inference_view(c);details=independent_weight(model,masked)
                    w=.5 if f.endswith('uniform') else details['private']
                    details.update(private=w,shared=1-w)
                    if f in ('cv','uniform'):errors['weight']=max(errors['weight'],max(abs(v-float(wl[c.id,f][k])) for k,v in details.items()))
                    shared=independent_predict(model,masked,'shared',point_mean=True)
                    private=independent_predict(model,masked,'private',point_mean=True)
                    q=shared+w*(private-shared)
                    correction=q
                    a=fold['choices'][f]['selected']['gain']
                    for g in parents:
                        expected[mapped(g+'4',f)]=(1-a)*expected[g]+a*q
                    expected['branch_'+f]=correction
                with np.load(root/tail) as z,np.load(control/tail) as old:
                    assert set(z.files)==set(expected)|{'y'} and np.array_equal(z['y'],y)
                    for g in parents:
                        for family in ('cv','uniform'):
                            oldkey=g+'4'+('_uniform' if family=='uniform' else '')
                            errors['control']=max(errors['control'],float(np.max(abs(z[mapped(g+'4','original_'+family)]-old[oldkey]))))
                    for g,p in expected.items():
                        errors['prediction']=max(errors['prediction'],float(np.max(abs(z[g]-p))))
                        if g.startswith('branch_'):continue
                        e=(np.asarray(p,float)-np.asarray(y,float))*100
                        values=np.array([np.mean(abs(e)),np.sqrt(np.mean(e*e)),np.quantile(abs(e),.95)])
                        row=lookup[c.id,g];assert row['dataset']==c.dataset and row['domain']==c.domain
                        errors['metric']=max(errors['metric'],float(np.max(abs(values-[float(row[k]) for k in ('mae','rmse','p95_ae')]))))
                        buckets[c.dataset,g,c.domain].append(values);cm[c.id,g]=(c.dataset,c.domain,values);seen.add((c.id,g))
            print(fold['fold'],'independent source-selection replay',flush=True)
    assert seen==set(lookup)
    means=defaultdict(list)
    for (ds,g,_),values in buckets.items():means[ds,g].append(np.mean(values,axis=0))
    table={key:np.mean(values,axis=0) for key,values in means.items()};assert len(table)==len(summary['table'])==120
    for r in summary['table']:errors['metric']=max(errors['metric'],float(np.max(abs(table[r['dataset'],r['group']]-[r[k] for k in ('mae','rmse','p95_ae')]))))
    assert max(errors.values())<1e-8,errors
    edges=[]
    for n in range(4):
        for subset in itertools.combinations('1234',n):
            for add in sorted(set('1234')-set(subset)):
                edges.append(('B'+''.join(sorted((*subset,add))),'B'+''.join(subset)))
    assert len(edges)==32;comparisons=[]
    for f in families:
        gates={a+'<'+b:bool(all(np.all(table[ds,mapped(a,f)]<table[ds,mapped(b,f)]) for ds in datasets)) for a,b in edges}
        assert gates==summary['gates'][f]
        comparisons.extend((f,mapped(a,f),mapped(b,f)) for a,b in edges)
    for f in ('cv','uniform'):
        comparisons.extend(('selection_control',mapped(g,f),mapped(g,'original_'+f)) for g in ('B4','B1234'))
    comparisons.extend(('weight_control',g,mapped(g,'uniform')) for g in ('B4','B1234'))
    pairs=[]
    for family,a,b in comparisons:
        for ds in datasets:
            ids=sorted(cid for cid,g in cm if g==a and cm[cid,g][0]==ds)
            for i,k in enumerate(('mae','rmse','p95_ae')):
                delta=[(cm[cid,a][2][i]-cm[cid,b][2][i])/100 for cid in ids]
                domains=[cm[cid,a][1] for cid in ids]
                pairs.append(dict(family=family,comparison=a+'-'+b,dataset=ds,metric=k,**paired_stats(delta,domains)))
        print(family,a,b,'paired intervals',flush=True)
    assert len(pairs)==1206
    sa.write_json(root/'verification.json',dict(status='PASS',rows=14600,edges=128,errors=errors,paired_comparisons=pairs,
        summary_sha256=sa.digest(root/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        helper_hashes={n:sa.digest(HERE/n) for n in ('audit_support_cv.py','audit_conditional_mixed.py')},
        limits='Independent target arithmetic on saved audited source GPs; no full retraining. Intervals uncorrected for repeated development selection. Source selection inherits historical parent choices. Optimizer warnings remain; no independent generalization claim.'))
    print('PASS14600rows128edges1206pairs',errors,flush=True)


if __name__=='__main__':main()

