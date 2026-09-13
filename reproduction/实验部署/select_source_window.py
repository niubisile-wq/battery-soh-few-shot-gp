"""Select a relative window on paired source-validation observations only."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'开发基线选择依据'))
from run_remaining_baselines import features


def main():
    base=ROOT/'实验部署/source_window_cache_v2'
    manifest=json.loads((base/'manifest.json').read_text())
    if manifest['status']!='SOURCE_CACHES_READY_SELECTION_PENDING':
        raise RuntimeError('Source cache build incomplete')
    keys=list(manifest['candidates'])
    data={key: {'source_train':[], 'source_validation':[]} for key in keys}
    matched=[]
    reference_flags=[]
    for cell,split in manifest['split'].items():
        paths=[base/key/split/(Path(cell).stem+'.npz') for key in keys]
        if not all(p.exists() for p in paths):
            raise RuntimeError(f'Candidate has no usable cell: {cell}')
        arrays=[np.load(p) for p in paths]
        common=sorted(set.intersection(*(set(a['cycle_number'].tolist()) for a in arrays)))
        if len(common)<=10:
            raise RuntimeError(f'Insufficient common early support and query: {cell}')
        matched.append({'cell_id':cell,'split':split,'common_cycles':len(common),
                        'support_cycle_numbers':common[:10]})
        for key,a in zip(keys,arrays):
            lookup={n:i for i,n in enumerate(a['cycle_number'])}
            indices=[lookup[n] for n in common]
            cap=a['capacity_Ah'][indices]
            nominal=float(a['nominal_capacity_Ah'])
            # Sanity flag only: never delete points or select later labels
            # to improve validation error. Dataset-specific reference audit
            # must resolve flagged early reference measurements first.
            if not .5 <= cap[0]/nominal <= 1.5:
                reference_flags.append({'cell_id':cell,'candidate':key,
                                        'first_capacity_Ah':float(cap[0]),
                                        'nominal_capacity_Ah':nominal,
                                        'early_support_capacities_Ah':cap[:10].tolist()})
            y=cap/cap[0]  # first reliable common support capacity fallback
            h=features(a['x'][indices])
            data[key][split].append((cell,h,y))
    # Fixed sampling indices, identical cycle identities across candidates.
    scores=[]
    for key in keys:
        tr=data[key]['source_train'];va=data[key]['source_validation']
        x=np.concatenate([h for _,h,_ in tr]);y=np.concatenate([y for _,_,y in tr])
        keep=np.linspace(0,len(y)-1,min(1000,len(y))).astype(int)
        model=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(x[keep],y[keep])
        cells=[]
        for cell,h,y in va:
            pred=model.predict(h[10:])
            cells.append({'cell_id':cell,'query_count':len(y)-10,
                          'mae':float(mean_absolute_error(y[10:],pred))})
        scores.append({'candidate':key,'macro_mae':float(np.mean([c['mae'] for c in cells])),
                       'validation_cells':cells})
    winner=min(scores,key=lambda r:(r['macro_mae'],r['candidate']))
    report={'status':'REFERENCE_AUDIT_REQUIRED' if reference_flags else 'WINDOW_SELECTED_ON_SOURCE_VALIDATION_ONLY',
            'selected':None if reference_flags else winner['candidate'],
            'relative_offsets_V':None if reference_flags else manifest['candidates'][winner['candidate']],
            'reference_flags':reference_flags,
            'scores':scores,'matched_observations':matched,'target_archives_read':[],
            'selector':'Ridge alpha=1, fixed 1000 paired source observations, macro validation MAE',
            'label_policy':'first reliable common early-support capacity fallback',
            'cache_manifest_sha256':hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),
            'limitations':['linear HI selector; neural convergence validation still required',
                           'target compatibility and complete baseline reruns still required']}
    (base/'window_selection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'selected':report['selected'],'reference_flags':len(reference_flags),'scores':[(s['candidate'],s['macro_mae']) for s in scores]},indent=2))


if __name__=='__main__':main()
