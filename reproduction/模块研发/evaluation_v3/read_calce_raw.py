"""Lossless numeric staging of prospective CALCE records, without model scoring.

No filtering by capacity values, no fitting, no cycle merging in this stage.
One compressed artifact per workbook supports reproducible chronology audits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import openpyxl
from openpyxl.utils.datetime import from_excel
from evaluate import sa, HERE

COLUMNS=['Data_Point','Test_Time(s)','Date_Time','Cycle_Index',
         'Current(A)','Voltage(V)','Discharge_Capacity(Ah)']


def numeric_date(value):
    if value is None:return np.nan
    if isinstance(value,(int,float)):
        value=from_excel(value)
    if isinstance(value,str):value=datetime.fromisoformat(value)
    if not isinstance(value,datetime):raise ValueError('Unrecognized acquisition Date_Time')
    # Preserve the instrument's naive wall-clock ordering, not claim UTC provenance.
    if value.tzinfo is not None:raise ValueError('Mixed time-zone metadata requires review')
    return (value-datetime(1970,1,1)).total_seconds()


def read_book(task):
    cid,archive,member,target=task;target=Path(target)
    with zipfile.ZipFile(archive) as z: raw=z.read(member)
    source_sha=hashlib.sha256(raw).hexdigest()
    workbook=openpyxl.load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
    arrays=[];sheet_records=[]
    for sheet in workbook:
        if not sheet.title.startswith('Channel_'):continue
        it=sheet.values;header=next(it)
        if not set(COLUMNS)<=set(header):raise ValueError(f'Missing columns: {member}/{sheet.title}')
        cols=[header.index(k) for k in COLUMNS];rows=[];blank=0
        for row in it:
            if not any(v is not None for v in row):blank+=1;continue
            values=[row[j] if j<len(row) else None for j in cols]
            values[2]=numeric_date(values[2])
            rows.append([float(v) if v is not None else np.nan for v in values])
        a=np.asarray(rows,dtype=np.float64).reshape(-1,len(COLUMNS));arrays.append(a)
        sheet_records.append(dict(sheet=sheet.title,rows=len(a),blank_rows=blank))
    workbook.close()
    if not arrays:raise ValueError('No channel sheets')
    data=np.concatenate(arrays)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,records=data,columns=np.array(COLUMNS))
    result=dict(cell_id=cid,member=member,source_sha256=source_sha,cache=str(target),
        cache_sha256=sa.digest(target),rows=len(data),sheets=sheet_records,
        nonfinite_rows=int((~np.isfinite(data).all(1)).sum()),
        wallclock_backward_steps=int((np.diff(data[:,2])<0).sum()),
        test_time_nonincreasing_steps=int((np.diff(data[:,1])<=0).sum()),
        first_wallclock=float(data[0,2]) if np.isfinite(data[0,2]) else None,
        last_wallclock=float(data[-1,2]) if np.isfinite(data[-1,2]) else None)
    sa.write_json(target.with_suffix('.json'),result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol_path=HERE/'calce_cohort_protocol.json';protocol=json.loads(protocol_path.read_text())
    audit_path=sa.ROOT/'模块研发/results/calce_unused_cohort_audit_v1/audit.json'
    audit=json.loads(audit_path.read_text());byid={r['cell_id']:r for r in audit['candidates']}
    source_path=sa.ROOT/'模块研发/results/calce_cohort_schema_audit_v1/audit.json'
    sources=json.loads(source_path.read_text())
    assert sources['protocol_sha256']==sa.digest(protocol_path)
    assert all(sa.digest(sa.ROOT/r['path'])==r['sha256'] for r in sources['source_artifacts_locked'])
    tasks=[]
    for cid in protocol['primary_candidates']:
        r=byid[cid];assert r['exposure_status']=='NO_MATCH_IN_AUDITED_HISTORY'
        archive=sa.ROOT/r['archive'];assert sa.digest(archive)==r['sha256']
        for member in r['members']:
            if member.endswith('.xlsx'):
                key=hashlib.sha256(member.encode()).hexdigest()[:16]
                tasks.append((cid,str(archive),member,str(out/'files'/cid/(key+'.npz'))))
    request=dict(status='RUNNING',purpose='Raw staging only; no predictions or target-based model selection',
        expected_workbooks=len(tasks),protocol_sha256=sa.digest(protocol_path),
        code_sha256=sa.digest(Path(__file__)),exposure_audit_sha256=sa.digest(audit_path),
        source_lock_sha256=sa.digest(source_path))
    sa.write_json(out/'request.json',request);rows=[];start=time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(read_book,t) for t in tasks]
            for future in as_completed(futures):
                row=future.result();rows.append(row)
                print(f"{len(rows)}/{len(tasks)} {row['cell_id']} {row['member']} rows={row['rows']} nonfinite={row['nonfinite_rows']}",flush=True)
        summary=dict(status='RAW_STAGING_COMPLETE_NOT_ELIGIBILITY_APPROVAL',workbooks=len(rows),
            raw_rows=sum(r['rows'] for r in rows),elapsed_seconds=time.monotonic()-start,
            files=sorted(rows,key=lambda r:(r['cell_id'],r['member'])),
            limits=['Numeric values retained including nonfinite values; not silently filtered.',
                    'File-local cycle indices not treated as global physical cycle numbers.',
                    'No overlap removal, cycle-label pairing or target model evaluation yet.'])
        sa.write_json(out/'summary.json',summary)
        request.update(status='COMPLETE',completed_workbooks=len(rows));sa.write_json(out/'request.json',request)
    except Exception as exc:
        request.update(status='FAILED',completed_workbooks=len(rows),error=repr(exc))
        sa.write_json(out/'request.json',request);raise


if __name__=='__main__':main()
