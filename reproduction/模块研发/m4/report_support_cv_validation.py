"""Describe audited validation choices/weights without any query metrics."""
import json
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa,HERE
from evaluate import dump_csv


def main():
    root=HERE.parent/'results/m4_support_cv_validation_v1'
    audit=json.loads((root/'verification.json').read_text())
    assert audit['status']=='PASS' and audit['result_sha256']==sa.digest(root/'result.json')
    result=json.loads((root/'result.json').read_text());rows=[];choices=[]
    for entry in result['selections']:
        folder=Path(entry['root']);assert sa.digest(folder/'result.json')==entry['result_sha256']
        fold=json.loads((folder/'result.json').read_text())
        assert sa.digest(folder/'predictions.joblib')==fold['prediction_sha256']
        saved=joblib.load(folder/'predictions.joblib');selected=fold['selected']['cv'];control=fold['selected']['uniform']
        zero=next(t['validation_mae'] for t in fold['trials'] if t['key']==selected['key'] and t['gain']==0)
        choices.append(dict(fold=entry['fold'],cv_key=selected['key'],cv_gain=selected['gain'],
            uniform_key=control['key'],uniform_gain=control['gain'],
            cv_delta_parent_pp=100*(selected['validation_mae']-zero),
            cv_delta_uniform_pp=100*(selected['validation_mae']-control['validation_mae'])))
        for (cid,kernel),w in saved['weights'].items():
            rows.append(dict(fold=entry['fold'],dataset=entry['fold'].split(':')[0],cell_id=cid,kernel=kernel,
                selected_kernel=selected['key']=='cv__'+kernel,selected_gain=selected['gain'],**w))
    assert len(choices)==21
    stats=[]
    for ds in ('XJTU','MATR','Tongji'):
        for kernel in ('rbf','matern'):
            subset=[r for r in rows if r['dataset']==ds and r['kernel']==kernel]
            ww=np.array([r['private'] for r in subset])
            stats.append(dict(dataset=ds,kernel=kernel,validation_episodes=len(ww),
                private_weight_mean=float(ww.mean()),private_weight_quantiles=np.quantile(ww,[0,.25,.5,.75,1]).tolist(),
                exactly_zero=int(np.sum(ww==0)),exactly_one=int(np.sum(ww==1))))
    out=HERE.parent/'results/m4_support_cv_validation_diagnosis_v1';out.mkdir(exist_ok=False)
    dump_csv(out/'weights.csv',rows);dump_csv(out/'choices.csv',choices)
    sa.write_json(out/'summary.json',dict(status='VALIDATION_ONLY_DESCRIPTIVE_REPORT',weights=len(rows),stats=stats,
        zero_gain_folds=[r['fold'] for r in choices if r['cv_gain']==0],
        source_result_sha256=sa.digest(root/'result.json'),source_audit_sha256=sa.digest(root/'verification.json'),
        code_sha256=sa.digest(Path(__file__)),
        limits='Repeated validation episodes,not unique independent cells. No query truth read,no new settings chosen. Weight extremity is descriptive,not calibrated accuracy or overfitting proof.'))
    print('weights',len(rows),'zero_gain',[r['fold'] for r in choices if r['cv_gain']==0],flush=True)
    print(stats,flush=True)


if __name__=='__main__':main()
