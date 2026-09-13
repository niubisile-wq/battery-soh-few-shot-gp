"""Source-only development caches for fixed relative-cutoff candidates.

No target archive is read. Keep physical seconds, capacity in Ah and actual
cycle IDs; labels are normalized later using eligible early support only.
"""
import collections
import hashlib
import json
from pathlib import Path
import pickle
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'开发基线选择依据'))
from partial_charge_protocol import Window, extract
from capacity_eligibility import xjtu_capacity


def main():
    out = ROOT/'实验部署/source_window_cache_v2'
    out.mkdir(exist_ok=True)
    archive = ROOT/'数据集/BatteryLife_v12_processed/XJTU.zip'
    # Offsets below the recorded hardware/protocol cutoff; never a lifetime
    # maximum estimated from held-out signals. Candidate choice remains open.
    candidates = {'offset_04_02': (.4, .2), 'offset_03_01': (.3, .1)}
    errors = collections.Counter()
    records = []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.pkl'))
        split = {n: ('source_validation' if i % 5 == 4 else 'source_train')
                 for i, n in enumerate(names)}
        manifest = {'status': 'BUILDING', 'split': split, 'candidates': candidates,
                    'capacity_eligibility': 'XJTU explicit RPT if present, otherwise measured full discharge to recorded cutoff',
                    'eligibility_sha256': hashlib.sha256((ROOT/'开发基线选择依据/capacity_eligibility.py').read_bytes()).hexdigest(),
                    'capacity_unit': 'Ah', 'signal_units': ['V','A','seconds'],
                    'label_normalization': 'deferred to support/reference policy',
                    'target_archives_read': [],
                    'extractor_sha256': hashlib.sha256((ROOT/'开发基线选择依据/partial_charge_protocol.py').read_bytes()).hexdigest()}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        for index, name in enumerate(names):
            with z.open(name) as stream:
                obj = pickle.load(stream)
            cutoff = float(obj['max_voltage_limit_in_V'])
            nominal = float(obj['nominal_capacity_in_Ah'])
            cycles = sorted(obj['cycle_data'], key=lambda c: float(c['cycle_number']))
            has_rpt = any(c.get('attribute') == 'RPT' for c in cycles)
            ids = [float(c['cycle_number']) for c in cycles]
            if len(set(ids)) != len(ids) or not np.isfinite(ids).all():
                raise ValueError(f'Ambiguous cycle identity: {name}')
            for key, (lower, upper) in candidates.items():
                window = Window(cutoff-lower, cutoff-upper)
                xs, caps, nums = [], [], []
                for c in cycles:
                    try:
                        capacity = xjtu_capacity(c, obj, has_rpt)
                        x, meta = extract(c, window)
                        xs.append(x); caps.append(capacity); nums.append(meta['cycle_number'])
                    except (ValueError, KeyError) as exc:
                        errors[(name, key, str(exc))] += 1
                target = out/key/split[name]
                target.mkdir(parents=True, exist_ok=True)
                path = target/(Path(name).stem+'.npz')
                if xs:
                    np.savez_compressed(path, x=np.stack(xs), capacity_Ah=np.asarray(caps),
                                        cycle_number=np.asarray(nums), nominal_capacity_Ah=nominal,
                                        cutoff_voltage_V=cutoff, cell_id=name)
                records.append({'cell_id': name, 'split': split[name], 'candidate': key,
                                'valid_cycles': len(xs), 'total_cycles': len(cycles),
                                'path': str(path.relative_to(out)) if xs else None})
            print(f'{index+1}/{len(names)} {name}', flush=True)
        manifest.update(status='SOURCE_CACHES_READY_SELECTION_PENDING', cells=records,
                        rejection_counts=[dict(cell_id=k[0], candidate=k[1], reason=k[2], count=v)
                                          for k,v in errors.items()])
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps({'cells':len(names), 'cache_records':len(records), 'status':manifest['status']}))


if __name__ == '__main__':
    main()
