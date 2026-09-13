"""Pre-score exposure audit and lossless raw staging of remaining CALCE cells."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile
import numpy as np
import openpyxl
import xlrd
from openpyxl.utils.datetime import from_excel
from study import HERE, ROOT, BUNDLE, PROTOCOL, PROTOCOL_PATH, digest, write_json, preservation

OUT=HERE/'external'
COLUMNS=['Data_Point','Test_Time(s)','Date_Time','Cycle_Index','Current(A)','Voltage(V)','Discharge_Capacity(Ah)']


def audit():
    preservation()
    dest=OUT/'exposure_and_source_lock.json'
    if dest.exists():
        a=json.loads(dest.read_text())
        assert a['protocol_sha256']==digest(PROTOCOL_PATH)
        for p,h in a['source_models'].items():assert digest(p)==h
        for r in a['archives']:assert digest(r['archive'])==r['sha256']
        return a
    ids=PROTOCOL['external_confirmation']['candidate_ids']
    roots=[ROOT/'模块研发/results',ROOT/'开发基线选择依据/results',ROOT/'实验部署']
    regex=r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(cid) for cid in ids)+r')(?![A-Za-z0-9])'
    cmd=['rg','-n','-P',regex,*map(str,roots),'-g','*.csv','-g','*.json',
         '-g','!**/source_snapshot/**','-g','!**/audit_snapshot/**',
         '-g','!**/calce_unused_cohort_audit_v1/**','-g','!**/calce_cohort_schema_audit_v1/**']
    found=subprocess.run(cmd,capture_output=True,text=True)
    assert found.returncode in [0,1],found.stderr
    archives=[]
    for cid in ids:
        path=ROOT/'数据集/CALCE'/cid[:3]/(cid+'.zip')
        with zipfile.ZipFile(path) as z:
            members=[dict(member=i.filename,size=i.file_size,crc=i.CRC) for i in z.infolist() if not i.is_dir()]
        archives.append(dict(cell_id=cid,archive=str(path),sha256=digest(path),members=members))
    manifest=json.loads((BUNDLE/'candidate.json').read_text())
    sources={str(BUNDLE/r['artifact']):r['sha256'] for r in manifest['manifest']
             if r['fold'].startswith('XJTU:') and r['group'] in ['B4','B14','B124','B1234']}
    assert len(sources)==24
    for p,h in sources.items():assert digest(p)==h
    a=dict(created_unix=time.time(),protocol_sha256=digest(PROTOCOL_PATH),search_command=cmd,
        historical_score_matches=found.stdout.splitlines(),source_models=sources,archives=archives,
        status='NO_MATCH_IN_AUDITED_RESULT_HISTORY' if not found.stdout.strip() else 'REVIEW_REQUIRED',
        previous_metadata_audit_sha256=digest(ROOT/'模块研发/results/calce_unused_cohort_audit_v1/audit.json'),
        official_description='https://web.calce.umd.edu/batteries/data/',
        scope='Only audited local results, not global never-use. CALCE source previously seen; these candidate cells were not found in old score manifests. Raw file inventory and source models locked before new model evaluation.')
    write_json(dest,a)
    assert a['status']=='NO_MATCH_IN_AUDITED_RESULT_HISTORY',a['historical_score_matches']
    return a


def date_number(v,datemode=0):
    if v is None:return np.nan
    if isinstance(v,(float,int)):v=xlrd.xldate_as_datetime(v,datemode)
    if isinstance(v,str):
        try:v=datetime.fromisoformat(v)
        except ValueError:
            for fmt in ['%m/%d/%Y %H:%M:%S','%m/%d/%Y %H:%M:%S.%f']:
                try:v=datetime.strptime(v,fmt);break
                except ValueError:pass
    if not isinstance(v,datetime) or v.tzinfo is not None:raise ValueError('Ambiguous acquisition date')
    return (v-datetime(1970,1,1)).total_seconds()


def read_excel(raw):
    arrays=[];sheets=[]
    if raw.startswith(b'PK'):
        book=openpyxl.load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
        iterator=((s.title,s.values) for s in book)
        datemode=0
    else:
        book=xlrd.open_workbook(file_contents=raw,on_demand=True)
        iterator=((s.name,(s.row_values(i) for i in range(s.nrows))) for s in book.sheets())
        datemode=book.datemode
    for name,rows in iterator:
        it=iter(rows);header=next(it,None)
        if header is None or not set(COLUMNS)<=set(header):
            sheets.append(dict(sheet=name,status='NO_MEASUREMENT_SCHEMA'));continue
        columns=[list(header).index(k) for k in COLUMNS];values=[];blank=0
        for row in it:
            if not any(v is not None and v!='' for v in row):blank+=1;continue
            vals=[row[j] if j<len(row) else None for j in columns]
            vals[2]=date_number(vals[2],datemode)
            values.append([float(v) if v is not None and v!='' else np.nan for v in vals])
        a=np.asarray(values,dtype=float).reshape(-1,7);arrays.append(a)
        sheets.append(dict(sheet=name,status='MEASUREMENTS',rows=len(a),blank_rows=blank))
    if hasattr(book,'close'):book.close()
    else:book.release_resources()
    if not arrays:raise ValueError('No standard measurement sheet; manual schema review required')
    return np.concatenate(arrays),sheets


def read_text(raw):
    h=raw.splitlines()[0].decode('utf-8-sig').split('\t')
    cols=['Time','Pgm step','Pgm cycle','mV','mA','Duration','Capacity']
    if not set(cols)<=set(h):raise ValueError('Unknown TXT schema')
    a=np.loadtxt(io.BytesIO(raw),skiprows=1,delimiter='\t',usecols=[h.index(c) for c in cols],ndmin=2)
    dt=np.diff(a[:,0]);duration=np.diff(a[:,5]);ok=(dt>0)&(duration>0)&(duration<1000)
    ratio=float(np.median(duration[ok]/dt[ok])) if ok.any() else None
    # Keep raw TXT numbers and labels. No guessed capacity/time conversion here.
    return a,dict(columns=cols,time_duration_median_ratio=ratio,
        unit_status='RAW_UNITS_REQUIRE_PHYSICAL_AND_SCHEMA_VALIDATION',
        chronology='Filename is download date, not a precise acquisition timestamp; no cross-file joining permitted')


def stage(task):
    cid,archive,member=task
    key=hashlib.sha256(member.encode()).hexdigest()[:20]
    target=OUT/'raw'/cid/(key+'.npz');meta=target.with_suffix('.json')
    if meta.exists():
        r=json.loads(meta.read_text())
        if r.get('status')=='STAGED':assert digest(target)==r['cache_sha256']
        return r
    with zipfile.ZipFile(archive) as z:raw=z.read(member)
    row=dict(cell_id=cid,archive=archive,member=member,source_sha256=hashlib.sha256(raw).hexdigest())
    try:
        if Path(member).suffix.lower() in ['.xlsx','.xls']:
            a,details=read_excel(raw);schema='arbin';columns=COLUMNS
        else:
            a,details=read_text(raw);schema='txt';columns=details['columns']
        target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,records=a,columns=np.array(columns))
        row.update(status='STAGED',schema=schema,cache=str(target),cache_sha256=digest(target),rows=len(a),
                   nonfinite_rows=int((~np.isfinite(a).all(1)).sum()),details=details)
    except Exception as exc:
        row.update(status='FAILED_SCHEMA_OR_READ',error=repr(exc))
    write_json(meta,row)
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--audit-only',action='store_true')
    args=ap.parse_args();a=audit()
    print('Locked',len(a['archives']),'candidate archives and',len(a['source_models']),'frozen source artifacts',flush=True)
    if args.audit_only:return
    tasks=[];ancillary=[]
    for r in a['archives']:
        for f in r['members']:
            ext=Path(f['member']).suffix.lower()
            if ext in ['.xlsx','.xls','.txt']:
                tasks.append((r['cell_id'],r['archive'],f['member']))
            else:ancillary.append(dict(cell_id=r['cell_id'],**f))
    write_json(OUT/'staging_request.json',dict(created_unix=time.time(),tasks=tasks,ancillary=ancillary,
        code_sha256=digest(Path(__file__)),source_lock_sha256=digest(OUT/'exposure_and_source_lock.json'),
        purpose='Raw staging and data integrity only. No model predictions.'))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(stage,t) for t in tasks]):
            r=f.result();rows.append(r)
            if len(rows)%10==0 or r['status']!='STAGED':
                print('STAGED',len(rows),'/',len(tasks),r['cell_id'],r['member'],r['status'],flush=True)
                write_json(OUT/'staging_progress.json',dict(completed=len(rows),total=len(tasks),
                    failures=[r for r in rows if r['status']!='STAGED']))
    write_json(OUT/'staging_summary.json',dict(status='RAW_STAGING_DONE_NOT_ELIGIBILITY_APPROVAL',
        rows=rows,total=len(tasks),failures=sum(r['status']!='STAGED' for r in rows),
        model_predictions_generated=False))
    print('STAGING_DONE',len(rows),'failures',sum(r['status']!='STAGED' for r in rows),flush=True)


if __name__=='__main__':main()
