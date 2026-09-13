"""Inspect prospective CALCE workbooks without extracting capacity trajectories."""
import io
import json
from pathlib import Path
import zipfile
from concurrent.futures import ProcessPoolExecutor
import openpyxl
from evaluate import sa, HERE


def inspect_cell(cid):
    archive=sa.ROOT/'数据集/CALCE'/cid.split('_')[0]/(cid+'.zip')
    required=json.loads((HERE/'calce_cohort_protocol.json').read_text())['signal_schema']
    files=[]
    with zipfile.ZipFile(archive) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('.xlsx')):
            workbook=openpyxl.load_workbook(io.BytesIO(z.read(name)),read_only=True,data_only=True)
            sheets=[]
            for sheet in workbook:
                if not sheet.title.startswith('Channel_'):continue
                iterator=sheet.values;header=next(iterator)
                first=next(iterator,None)
                missing=[k for k in required if k not in header]
                # First-row acquisition identifiers only; never inspect capacity values.
                identity={k:str(first[header.index(k)]) for k in ['Data_Point','Test_Time(s)','Date_Time','Cycle_Index']
                          if first is not None and k in header}
                sheets.append(dict(sheet=sheet.title,reported_rows=sheet.max_row,header=list(header),
                                   missing_required_columns=missing,first_record_identifiers=identity))
            workbook.close()
            files.append(dict(member=name,channel_sheets=sheets))
        unhandled=[n for n in z.namelist() if not n.endswith('/') and not n.endswith('.xlsx')]
    return dict(cell_id=cid,archive=str(archive.relative_to(sa.ROOT)),archive_sha256=sa.digest(archive),
        files=files,other_members=unhandled,
        schema_pass=bool(files) and all(f['channel_sheets'] and all(not s['missing_required_columns']
            for s in f['channel_sheets']) for f in files))


def main():
    out=sa.ROOT/'模块研发/results/calce_cohort_schema_audit_v1'
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((HERE/'calce_cohort_protocol.json').read_text())
    sa.write_json(out/'request.json',dict(status='RUNNING',protocol_sha256=sa.digest(HERE/'calce_cohort_protocol.json')))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(inspect_cell,protocol['primary_candidates']))
    source=sa.ROOT/'模块研发/results/m2_reference_candidate_v3'
    candidate=json.loads((source/'candidate.json').read_text())
    models=[]
    for r in candidate['manifest']:
        if not r['fold'].startswith('XJTU:'):continue
        path=source/r['artifact'];assert sa.digest(path)==r['sha256']
        models.append(dict(path=str(path.relative_to(sa.ROOT)),sha256=r['sha256']))
    for fold in sorted({r['fold'] for r in candidate['manifest'] if r['fold'].startswith('XJTU:')}):
        job=sa.ROOT/'模块研发/results/m1_v1/screen_v1/jobs'/('P04_pls_metric__'+fold.replace(':','__'))
        for name in ['model.joblib','tuning.json','provenance.json']:
            p=job/name;models.append(dict(path=str(p.relative_to(sa.ROOT)),sha256=sa.digest(p)))
    report=dict(status='SCHEMA_AUDITED_CHRONOLOGY_AND_CAPACITY_ALIGNMENT_PENDING',
        code_sha256=sa.digest(Path(__file__)),protocol_sha256=sa.digest(HERE/'calce_cohort_protocol.json'),
        cells=rows,source_artifacts_locked=models,
        all_required_columns_present=all(r['schema_pass'] for r in rows),
        limits=['Headers and first acquisition identifiers only; no capacity trajectories or predictions evaluated.',
                'Reported worksheet dimensions may include blank rows; not an eligible cycle count.',
                'Full chronology, overlaps, cycle stitching and capacity reference still require a separate audit.'])
    sa.write_json(out/'audit.json',report)
    sa.write_json(out/'request.json',dict(status='COMPLETE',all_required_columns_present=report['all_required_columns_present']))
    print([(r['cell_id'],len(r['files']),r['schema_pass']) for r in rows],flush=True)


if __name__=='__main__':main()
