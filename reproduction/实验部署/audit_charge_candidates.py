"""Audit candidate window coverage on XJTU source-training cells only."""
import collections
import csv
import hashlib
import json
import pickle
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '开发基线选择依据'))
from partial_charge_protocol import Window, extract


def main():
    out = ROOT / '实验部署/charge_candidates_v1'
    out.mkdir(exist_ok=True)
    archive = ROOT / '数据集/BatteryLife_v12_processed/XJTU.zip'
    windows = [Window(3.8, 4.0), Window(3.9, 4.1), Window(4.0, 4.2)]
    totals = collections.Counter()
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.pkl'))
        train = [n for i, n in enumerate(names) if i % 5 != 4]
        val = [n for i, n in enumerate(names) if i % 5 == 4]
        manifest = {'status': 'RUNNING', 'source_training_cells': train,
                    'source_validation_cells_unread': val,
                    'candidates': [vars(w) for w in windows],
                    'selection': 'coverage audit only; predictive validation and domain compatibility still required',
                    'extractor_sha256': hashlib.sha256((ROOT / '开发基线选择依据/partial_charge_protocol.py').read_bytes()).hexdigest()}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        with (out/'cycle_coverage.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['cell_id', 'cycle_number', 'low_v', 'high_v', 'valid', 'duration_s', 'reason'])
            writer.writeheader()
            for index, name in enumerate(train):
                with z.open(name) as stream:
                    obj = pickle.load(stream)
                for cycle in sorted(obj['cycle_data'], key=lambda c: float(c['cycle_number'])):
                    for w in windows:
                        row = dict(cell_id=name, cycle_number=cycle['cycle_number'], low_v=w.low_v, high_v=w.high_v,
                                   valid=False, duration_s='', reason='')
                        try:
                            _, meta = extract(cycle, w)
                            row.update(valid=True, duration_s=meta['duration_s'])
                        except (ValueError, KeyError) as exc:
                            row['reason'] = str(exc)
                        totals[(w.low_v, w.high_v, row['valid'], row['reason'])] += 1
                        writer.writerow(row)
                f.flush()
                print(f'{index+1}/{len(train)} {name}', flush=True)
        summary = [dict(low_v=k[0], high_v=k[1], valid=k[2], reason=k[3], count=v)
                   for k,v in totals.items()]
        (out/'coverage_summary.json').write_text(json.dumps(summary, indent=2))
        manifest['status'] = 'COVERAGE_AUDITED_NOT_WINDOW_SELECTED'
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
