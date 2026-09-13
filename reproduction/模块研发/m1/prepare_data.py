"""Apply the already-fixed partial-charge protocol to MATR and Tongji."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import pickle
import sys
import zipfile
from pathlib import Path
import numpy as np
from data import ROOT,CACHE,K,digest,write_json

sys.path.insert(0,str(ROOT/'开发基线选择依据'))
from partial_charge_protocol import Window,extract
from capacity_eligibility import xjtu_capacity


def prepare(spec):
    dataset,name=spec
    stem=Path(name).stem
    out=CACHE/dataset
    record=out/'records'/f'{stem}.json'
    if record.exists():
        import json
        return json.loads(record.read_text())
    if dataset=='Tongji':
        archive=ROOT/'数据集/BatteryLife_v12_processed/Tongji.zip'
        with zipfile.ZipFile(archive) as z:
            raw=z.read(name)
        obj=pickle.loads(raw)
        domain=stem.rsplit('--',1)[0]
    elif dataset=='MATR':
        raw=(ROOT/'数据集/MATR_BatteryML_processed'/name).read_bytes()
        obj=pickle.loads(raw)
        domain=stem.split('c',1)[0].split('MATR_',1)[1]
    else:
        raise ValueError(dataset)
    sha=hashlib.sha256(raw).hexdigest();del raw
    cutoff=float(obj['max_voltage_limit_in_V']);window=Window(cutoff-.3,cutoff-.1)
    cycles=sorted(obj['cycle_data'],key=lambda c:float(c['cycle_number']))
    numbers=[float(c['cycle_number']) for c in cycles]
    if len(set(numbers))!=len(numbers) or not np.isfinite(numbers).all():
        raise ValueError(f'Ambiguous cycle numbers: {name}')
    has_rpt=any(c.get('attribute')=='RPT' for c in cycles)
    xs,caps,nums=[],[],[];reject=Counter()
    for c in cycles:
        try:
            q=xjtu_capacity(c,obj,has_rpt)
            x,meta=extract(c,window)
        except (ValueError,KeyError) as exc:
            reject[str(exc)]+=1;continue
        xs.append(x);caps.append(q);nums.append(meta['cycle_number'])
    path=out/'cells'/f'{stem}.npz'
    eligible=len(caps)>K
    if eligible:
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,x=np.stack(xs),capacity_Ah=np.asarray(caps,dtype='float32'),
                            cycle_number=np.asarray(nums,dtype='float32'),cell_id=f'{dataset}/{stem}.pkl',
                            dataset=dataset,domain=domain,nominal_capacity_Ah=float(obj['nominal_capacity_in_Ah']),
                            cutoff_voltage_V=cutoff,min_voltage_limit_V=float(obj['min_voltage_limit_in_V']))
    result={'cell_id':f'{dataset}/{stem}.pkl','dataset':dataset,'domain':domain,
            'eligible':eligible,'valid_cycles':len(caps),'total_cycles':len(cycles),
            'path':str(path.relative_to(out)) if eligible else None,'rejections':dict(reject),
            'source_sha256':sha,'support_cycle_numbers':nums[:K],
            'metadata':{k:obj.get(k) for k in ['cathode_material','anode_material','nominal_capacity_in_Ah','charge_protocol','discharge_protocol']}}
    write_json(record,result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',default='MATR,Tongji');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    for ds in args.datasets.split(','):
        if ds=='MATR':names=sorted(p.name for p in (ROOT/'数据集/MATR_BatteryML_processed').glob('*.pkl'))
        elif ds=='Tongji':
            with zipfile.ZipFile(ROOT/'数据集/BatteryLife_v12_processed/Tongji.zip') as z:names=sorted(n for n in z.namelist() if n.endswith('.pkl'))
        else:raise ValueError('Only new declared development caches allowed')
        manifest={'dataset':ds,'status':'BUILDING','expected_cells':len(names),
                  'capacity_policy':'Explicit RPT where present, otherwise measured full discharge to recorded cutoff; no interpolated labels',
                  'window_offsets':[.3,.1],'extractor_sha256':digest(ROOT/'开发基线选择依据/partial_charge_protocol.py'),
                  'eligibility_sha256':digest(ROOT/'开发基线选择依据/capacity_eligibility.py'),
                  'label_reference':'First eligible measured support capacity; capacity recovery retained'}
        write_json(CACHE/ds/'manifest.json',manifest)
        records=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(prepare,(ds,n)) for n in names]
            for f in as_completed(futures):
                r=f.result();records.append(r)
                print(f'{ds} {len(records)}/{len(names)} {r["cell_id"]} eligible={r["valid_cycles"]}',flush=True)
        manifest.update(status='COMPLETE',cells=sorted(records,key=lambda r:r['cell_id']),eligible_cells=sum(r['eligible'] for r in records))
        write_json(CACHE/ds/'manifest.json',manifest)


if __name__=='__main__':main()
