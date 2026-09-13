"""Paired risk comparisons against unchanged previous-version controls."""
import argparse
import json
from pathlib import Path
import shutil
import repair_diagnostics as rd


def paired_draws(old,new):
    def index(result):
        records={}
        for r in result['resamples']:
            key=(r['fold'],r['group'],r['seed'])
            assert key not in records
            assert len(r['source_ids'])==len(r['counts'])
            assert len(set(r['source_ids']))==len(r['source_ids'])
            records[key]=dict(zip(r['source_ids'],r['counts'],strict=True))
        return records
    a=index(old);b=index(new);assert a==b
    return len(a)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',required=True,type=Path);ap.add_argument('--new',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    paths=[args.old.resolve(),args.new.resolve()]
    data=[]
    for p in paths:
        assert json.loads((p/'aggregation_audit.json').read_text())['status']=='PASS'
        r=json.loads((p/'summary.json').read_text());assert r['status']=='COMPLETE';data.append(r)
    n=paired_draws(*data)
    old,new=[{(r['seed'],r['dataset'],r['group']):r for r in d['summary']} for d in data]
    assert old.keys()==new.keys();seeds=sorted({k[0] for k in old});rows=[];gates=[]
    for seed in seeds:
        current=[]
        for dataset in rd.sa.PROTOCOL['datasets']:
            for name,ng,og in [('full_vs_old_full','Base+M1+M2','Base+M1+M2'),('full_vs_old_m2','Base+M1+M2','Base+M2'),('m2_vs_old_m2','Base+M2','Base+M2')]:
                r=dict(seed=seed,dataset=dataset,comparison=name,**{k:float(100*(new[seed,dataset,ng][k]-old[seed,dataset,og][k])) for k in rd.KEYS})
                rows.append(r);current.append(r)
        gates.append(dict(seed=seed,**{name:all(r[k]<0 for r in current if r['comparison']==name for k in rd.KEYS) for name in ['full_vs_old_full','full_vs_old_m2','m2_vs_old_m2']}))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    rd.sa.write_json(out/'comparison.json',dict(status='COMPLETE',paired_group_draws=n,rows=rows,gates=gates,
        input_hashes={str(p/'summary.json'):rd.sa.digest(p/'summary.json') for p in paths},
        limits='Same source risk draws only; fixed GP development data, not independent confirmation or full retraining. Differences in SOH percentage points.'))
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    for name in ['full_vs_old_full','full_vs_old_m2','m2_vs_old_m2']:print(name,sum(r[name] for r in gates),'/',len(gates))


if __name__=='__main__':main()
