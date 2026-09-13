"""Correct proven GPR source-budget metadata in separate historical copies.

Never alters predictions or treats legacy full-charge results as corrected-
protocol reruns. Original CSVs remain intact for traceability.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    results=ROOT/'开发基线选择依据/results'
    files=[results/'remaining_baselines_10seed.csv']
    for folder in ['remaining_baselines','remaining_baselines_seed5_9']:
        files.extend(results/folder/name for name in ['remaining_results.csv','remaining_cell_results.csv'])
    out=results/'historical_budget_corrections';out.mkdir(exist_ok=True)
    audit=[]
    for path in files:
        with path.open() as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        count=0
        for row in rows:
            if row['model']=='GPR' and row['variant']=='source_only':
                if row['source_budget']!='1000':
                    raise ValueError(f'Unexpected historical budget in {path}')
                row['source_budget']='200';count+=1
        target=out/(path.parent.name+'__'+path.name)
        with target.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        audit.append({'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'corrected_copy':str(target),'changed_rows':count,'changed_field':'source_budget',
                      'old':1000,'new':200})
    report={'status':'HISTORICAL_METADATA_CORRECTED_NOT_RERUN',
            'evidence':'run_remaining_baselines.py: model_source_budgets GPR=200, model_keep used for fitting',
            'files':audit,'originals_preserved':True}
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'files':len(audit),'changed_rows':sum(a['changed_rows'] for a in audit),'status':report['status']}))


if __name__=='__main__':main()
