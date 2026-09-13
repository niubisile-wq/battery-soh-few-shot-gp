"""Independent prediction arithmetic, 9490 three-metric rows and64 edge checks."""
import csv,json,itertools,argparse
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from source_oof import sa,HERE
from summarize_strict import paired_stats


def metrics(y,p):
    e=(np.asarray(p)-np.asarray(y))*100
    return dict(mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.square(e).mean())),p95_ae=float(np.percentile(np.abs(e),95)))


def main():
    ap=argparse.ArgumentParser();variants=ap.add_mutually_exclusive_group();variants.add_argument('--mixed-effects',action='store_true');variants.add_argument('--conditional-mixed',action='store_true');variants.add_argument('--evidence',action='store_true');variants.add_argument('--hierarchical',action='store_true');variants.add_argument('--support-cv',action='store_true');args=ap.parse_args()
    variant='support_cv' if args.support_cv else 'hierarchical' if args.hierarchical else 'evidence' if args.evidence else 'conditional_mixed' if args.conditional_mixed else 'mixed_effects' if args.mixed_effects else 'difference'
    families=('cv','uniform') if args.support_cv else ('hierarchical','cell_only') if args.hierarchical else ('evidence','uniform') if args.evidence else ('private','shared','bias') if args.conditional_mixed else ('mixed','zero') if args.mixed_effects else ('difference','direct')
    control='cell_only' if args.hierarchical else 'uniform' if args.evidence or args.support_cv else 'zero' if args.mixed_effects else 'direct'
    expected_rows=12775 if args.conditional_mixed else 9490
    root=HERE.parent/'results'/('m4_'+variant+'_screen_v1');candidate=HERE.parent/'results/m3_waveform_candidate_v1'
    frozen=json.loads((root/'frozen_selections.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert sa.digest(candidate/'candidate.json')==frozen['parent_manifest_sha256']
    parents=['B'+''.join(map(str,s)) for n in range(4) for s in itertools.combinations((1,2,3),n)]
    dslist=('XJTU','MATR','Tongji');cells=[c for ds in dslist for c in sa.load_cells(ds)]
    rows=list(csv.DictReader((root/'cells.csv').open()));lookup={(r['cell_id'],r['group']):r for r in rows}
    assert len(lookup)==len(rows)==expected_rows
    pred_error=0.;metric_error=0.;aggregated={};seen=set();cell_metrics={};worst_prediction={};worst_metric={};weight_error=0.
    if args.evidence or args.support_cv:
        weight_rows=list(csv.DictReader((root/'weights.csv').open()))
        weight_lookup={(r['cell_id'],r['family']):r for r in weight_rows}
        assert len(weight_lookup)==len(weight_rows)==730
    with threadpool_limits(limits=1):
        for fold in frozen['selections']:
            if args.conditional_mixed or args.evidence or args.support_cv:
                branchroot=Path(fold['validation_root']);assert sa.digest(branchroot/'result.json')==fold['validation_result_sha256']
            else:
                branchroot=Path(fold['branch_root']);assert sa.digest(branchroot/'result.json')==fold['result_sha256']
            models={}
            for t,s in fold['choices'].items():
                assert sa.digest(Path(s['model']['path']))==s['model']['sha256']
                models[t]=joblib.load(s['model']['path'])
            _,_,test=sa.split(cells,fold['fold'])
            for c in test:
                tail=Path('folds')/fold['fold'].replace(':','__')/'predictions'/(c.id+'.npz')
                with np.load(candidate/tail) as old:
                    y=old['y'].copy();expected={g:old[g].copy() for g in parents}
                np.testing.assert_array_equal(y,c.y[sa.K:])
                for target,model in models.items():
                    if args.support_cv:
                        from audit_support_cv import independent_weight
                        from audit_conditional_mixed import independent_predict
                        details=independent_weight(model,sa.inference_view(c))
                        w=details['private'] if target=='cv' else .5
                        q=(1-w)*independent_predict(model,sa.inference_view(c),'shared',point_mean=True)+w*independent_predict(model,sa.inference_view(c),'private',point_mean=True)
                        expected_weights=dict(details,private=w,shared=1-w)
                        weight_error=max(weight_error,max(abs(v-float(weight_lookup[c.id,target][k])) for k,v in expected_weights.items()))
                    elif args.hierarchical:
                        from audit_conditional_mixed import independent_predict
                        q=.5*(independent_predict(model.parent,sa.inference_view(c),'shared',point_mean=True)+independent_predict(model.parent,sa.inference_view(c),'private',point_mean=True))
                    elif args.evidence:
                        from audit_evidence import independent_weight
                        from audit_conditional_mixed import independent_predict
                        w=independent_weight(model,sa.inference_view(c))[0] if target=='evidence' else .5
                        q=(1-w)*independent_predict(model,sa.inference_view(c),'shared',point_mean=True)+w*independent_predict(model,sa.inference_view(c),'private',point_mean=True)
                        weight_error=max(weight_error,abs(w-float(weight_lookup[c.id,target]['private'])))
                    elif args.conditional_mixed:
                        from audit_conditional_mixed import independent_predict
                        q=independent_predict(model,sa.inference_view(c),target,point_mean=True)
                    else:q=model.predict(sa.inference_view(c))
                    gain=fold['choices'][target]['selected']['gain']
                    for g in parents:
                        suffix=('_'+target if target!='private' else '') if args.conditional_mixed else ('_'+control if target==control else '')
                        key=g+'4'+suffix
                        expected[key]=(1-gain)*expected[g]+gain*q
                    expected['branch_'+target]=q
                with np.load(root/tail) as saved:
                    assert set(saved.files)==set(expected)|{'y'}
                    np.testing.assert_array_equal(saved['y'],y)
                    for g,p in expected.items():
                        pe=float(np.max(abs(p-saved[g])))
                        if pe>pred_error:worst_prediction=dict(fold=fold['fold'],cell_id=c.id,group=g,index=int(np.argmax(abs(p-saved[g]))),error=pe)
                        pred_error=max(pred_error,pe)
                        m=metrics(y,p);row=lookup[c.id,g];seen.add((c.id,g))
                        cell_metrics[c.id,g]=dict(m,dataset=c.dataset,domain=c.domain)
                        assert row['dataset']==c.dataset and row['domain']==c.domain
                        for k,v in m.items():
                            me=abs(v-float(row[k]))
                            if me>metric_error:worst_metric=dict(fold=fold['fold'],cell_id=c.id,group=g,metric=k,error=me)
                            metric_error=max(metric_error,me)
                            aggregated.setdefault((c.dataset,g,k),{}).setdefault(c.domain,[]).append(v)
            print(fold['fold'],'independent prediction replay',flush=True)
    assert seen==set(lookup)
    table={(r['dataset'],r['group']):r for r in summary['table']}
    assert len(table)==(105 if args.conditional_mixed else 78)
    calculated={}
    for (ds,g,k),domain in aggregated.items():
        value=float(np.mean([np.mean(v) for v in domain.values()]));calculated[ds,g,k]=value
        metric_error=max(metric_error,abs(value-table[ds,g][k]))
    edges=[]
    for count in range(4):
        for subset in itertools.combinations((1,2,3,4),count):
            for addition in set((1,2,3,4))-set(subset):
                a='B'+''.join(map(str,sorted((*subset,addition))));b='B'+''.join(map(str,subset));edges.append((a,b))
    assert len(edges)==32
    comparisons=[]
    for target in families:
        assert set(summary['gates'][target])=={a+'<'+b for a,b in edges}
        for a,b in edges:
            def mapped(g):
                suffix=('_'+target if target!='private' else '') if args.conditional_mixed else ('_'+control if target==control else '')
                return g+(suffix if '4' in g else '')
            ok=all(calculated[ds,mapped(a),k]<calculated[ds,mapped(b),k] for ds in dslist for k in ('mae','rmse','p95_ae'))
            assert ok==summary['gates'][target][a+'<'+b]
            comparisons.append((target,mapped(a),mapped(b)))
    for ctl in (('shared','bias') if args.conditional_mixed else (control,)):
        comparisons.extend([('target_control','B4','B4_'+ctl),('target_control','B1234','B1234_'+ctl)])
    if max(pred_error,metric_error,weight_error)>=1e-8:
        sa.write_json(root/('audit_numeric_failure_'+sa.digest(Path(__file__))[:12]+'.json'),dict(prediction_error=pred_error,metric_error=metric_error,
            worst_prediction=worst_prediction,worst_metric=worst_metric,
            tolerance=1e-8,code_sha256=sa.digest(Path(__file__)),status='NUMERIC_REPLAY_MISMATCH'))
        print('Numeric replay mismatch',pred_error,metric_error,flush=True)
    assert max(pred_error,metric_error,weight_error)<1e-8
    pairs=[]
    for family,a,b in comparisons:
        for ds in dslist:
            ids=sorted(cid for cid,g in cell_metrics if g==a and cell_metrics[cid,g]['dataset']==ds)
            for k in ('mae','rmse','p95_ae'):
                delta=[(cell_metrics[cid,a][k]-cell_metrics[cid,b][k])/100 for cid in ids]
                domains=[cell_metrics[cid,a]['domain'] for cid in ids]
                pairs.append(dict(family=family,comparison=a+'-'+b,dataset=ds,metric=k,**paired_stats(delta,domains)))
        print(family,a,b,'paired intervals complete',flush=True)
    assert len(pairs)==(900 if args.conditional_mixed else 594)
    sa.write_json(root/'verification.json',dict(status='PASS',rows=expected_rows,prediction_error=pred_error,metric_error=metric_error,
        edges=len(families)*32,weight_error=weight_error,paired_comparisons=pairs,summary_sha256=sa.digest(root/'summary.json'),code_sha256=sa.digest(Path(__file__)),
        helper_hashes={n:sa.digest(HERE/n) for n in ('audit_support_cv.py','audit_conditional_mixed.py')} if args.support_cv else {n:sa.digest(HERE/n) for n in ('audit_evidence.py','audit_conditional_mixed.py')} if args.evidence else {'audit_conditional_mixed.py':sa.digest(HERE/'audit_conditional_mixed.py')} if args.hierarchical else {},
        limits='Saved-model replay and independent metrics/edges. Source refits in separate per-fold audits. Development intervals without selection/multiplicity correction; risk stability not computed, not independent confirmation.'))
    print('PASS',expected_rows,'rows',len(families)*32,'edges',flush=True)


if __name__=='__main__':main()
