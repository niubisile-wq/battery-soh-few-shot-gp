"""Full-bank global source-risk routing under nested domain-excluded fits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import nested_signal_probe as ns
import reference_gate as rg
from diagnose_expert_capacity import capacity


def enrich(records,raw,models,byid,modes):
    result=[];raw_byid={e['cell_id']:e for e in raw}
    for e in records:
        c=byid[e['cell_id']];support=ns.rd.sa.prefix(ns.rd.sa.inference_view(c),ns.rd.sa.K)
        residual={g:float(np.mean(support.y-ns.rd.sa.prior(m,support))) for g,m in models.items()}
        result.append(dict(e,residual=residual[modes],raw_base=raw_byid[c.id]['base'],raw_residual=residual['Base']))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);sa=ns.rd.sa
    root=sa.ROOT/'模块研发/results/m2_nested_signal_probe_v1'
    assert json.loads((root/'audit.json').read_text())['status']=='PASS'
    protocol=json.loads((root/'protocol.json').read_text());cells=sa.load_cells('MATR');rows=[];selections=[]
    for fold in protocol['folds']:
        train,_,_=sa.split(cells,fold);byid={c.id:c for c in train};folder=root/'folds'/fold.replace(':','__')
        result=json.loads((folder/'result.json').read_text());pairs={}
        for excluded in sorted({tuple(f['excluded']) for f in result['fits']}):
            fits=[f for f in result['fits'] if tuple(f['excluded'])==excluded]
            models={}
            for f in fits:
                if f['group']=='Physical_control': continue
                path=root/f['artifact'];assert sa.digest(path)==f['sha256'];models[f['group']]=joblib.load(path)
            episodes=joblib.load((root/fits[0]['artifact']).parent/'episodes.joblib')
            pairs[excluded]={p:enrich(episodes[p],episodes['Base'],models,byid,p) for p in episodes}
        original=joblib.load(folder/'source_lodo_episodes.joblib')
        for held in sorted({c.domain for c in train}):
            for parent in ['Base','Base+M1']:
                training=[]
                for domain in sorted({c.domain for c in train}-{held}):
                    training.extend(e for e in pairs[tuple(sorted([held,domain]))][parent] if e['domain']==domain)
                validation=[e for e in rg.with_raw_reference(original[parent],original['Base']) if e['domain']==held]
                assert not {e['cell_id'] for e in training}&{e['cell_id'] for e in validation}
                params=[dict(p,raw_mix=m) for m in ([0.] if parent=='Base' else [0.,.5,1.]) for p in rg.PARAMS]
                counts=defaultdict(int)
                for e in training: counts[e['domain']]+=1
                weights=np.array([1/len(counts)/counts[e['domain']] for e in training])
                risk=np.einsum('i,ijk->jk',weights,np.array([[rg.expert_risk(e,p) for p in params] for e in training]))
                for objective in ['mae','minimax']:
                    index=int(rg.risk_indices(risk,params,objective));p=params[index]
                    selections.append(dict(fold=fold,held=held,parent=parent,objective=objective,parameter=p,
                        train_ids=[e['cell_id'] for e in training],validation_ids=[e['cell_id'] for e in validation],risks=risk.tolist()))
                    for e in validation:
                        m=sa.metrics(e['y'],rg.expert_prediction(e,p));o=capacity(e,params)
                        rows.append(dict(fold=fold,domain=held,parent=parent,objective=objective,cell_id=e['cell_id'],
                            selected_mae_pp=m['mae']*100,selected_p95_pp=m['p95_ae']*100,
                            cell_oracle_mae_pp=o['cell_oracle_mae_pp'],point_oracle_mae_pp=o['point_oracle_mae_pp']))
    summary=[]
    for parent in ['Base','Base+M1']:
        for objective in ['mae','minimax']:
            grouped=defaultdict(list)
            for r in rows:
                if r['parent']==parent and r['objective']==objective: grouped[r['fold'],r['domain']].append(r)
            summary.append(dict(parent=parent,objective=objective,**{k:float(np.mean([np.mean([r[k] for r in g]) for g in grouped.values()])) for k in
                ['selected_mae_pp','selected_p95_pp','cell_oracle_mae_pp','point_oracle_mae_pp']}))
    sa.write_json(out/'result.json',dict(status='COMPLETE',summary=summary,rows=rows,selections=selections,
        input_audit_sha256=sa.digest(root/'audit.json'),limits=['Source diagnostic, not v8 routing or outer performance.',
        'Frozen configurations inherited; GP fitting nested, configuration selection not nested.',
        'Oracle values use validation answers and are not deployable.']))
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
