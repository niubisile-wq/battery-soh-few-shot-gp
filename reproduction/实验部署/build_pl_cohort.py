"""Measurement-only PL expansion. No predictions, no merging operations."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from inspect_pl import load_cell,old_table
from measurement import events,repair_clock
from study import HERE,ROOT,digest,write_json

OUT=HERE/'external'


def exposure():
    target=OUT/'pl_exposure_audit.json'
    cmd=['rg','-n','-P',r'(?<![A-Za-z0-9])PL_?(11|12|13|14)(?![A-Za-z0-9])|SOC_0.?100.*(HalfC|2C)',
         *map(str,[ROOT/'模块研发/results',ROOT/'开发基线选择依据/results',ROOT/'实验部署']),
         '-g','*.csv','-g','*.json','-g','!**/source_snapshot/**','-g','!**/audit_snapshot/**']
    found=subprocess.run(cmd,capture_output=True,text=True)
    assert found.returncode in [0,1],found.stderr
    assert not found.stdout.strip(),found.stdout
    record=dict(created_unix=time.time(),search_command=cmd,matches=[],
        status='NO_MATCH_IN_AUDITED_RESULT_HISTORY',model_predictions_generated=False,
        limitation='Audited local result manifests only; not proof of never being used anywhere',
        protocol_sha256=digest(OUT/'pl_protocol.json'),source_lock_sha256=digest(OUT/'pl_source_lock.json'))
    if target.exists():
        prior=json.loads(target.read_text())
        assert prior['protocol_sha256']==record['protocol_sha256']
        return
    write_json(target,record)


def main():
    exposure();protocol=json.loads((OUT/'pl_protocol.json').read_text())
    dest=OUT/'pl_cohort';dest.mkdir(exist_ok=True)
    implementation=dict(created_before_model_prediction=True,protocol_sha256=digest(OUT/'pl_protocol.json'),
        code_sha256={str(HERE/p):digest(HERE/p) for p in ['build_pl_cohort.py','measurement.py','inspect_pl.py']},
        table_decoder='mat-io 1.0.0, raw_data=True; explicitly named MATLAB R2014 table properties',
        operation_boundary='Never join different operations, including charge/discharge separated into separate tables. Missing Data entries explicitly omitted; first10 eligible paired measurements, not first10 physical ageing cycles.',
        date_conversion='Recorded MATLAB serial date minus719529, multiplied86400; timezone unspecified',
        duplicate_rule='Deduplicate only identical complete8-column rows; preserve raw table and original row mapping',
        sampling_rule='Same signal-defined charge/full-discharge rule, PL measured cutoff<=2.77V; counter integral envelope plus0.1% nominal tolerance',
        eligibility='All four preregistered cells evaluated for measurement eligibility; no model scores used')
    write_json(dest/'implementation.json',implementation)
    reports=[];fingerprints={};cross=[]
    for cid in protocol['candidate_ids']:
        values,meta=load_cell(cid);all_events=[];issues=[];omitted=[];raw_files=[];reject=Counter()
        for operation,row in enumerate(values[1:],1):
            name=str(np.asarray(row[0]).item());start=str(np.asarray(row[1]).item())
            spec=dict(operation_index=operation,operation=name,start_date_description=start)
            if not hasattr(row[2],'properties'):
                omitted.append(dict(spec,reason='NO_RECORDED_TABLE',raw_description=str(row[2])));continue
            original,columns=old_table(row[2]);n=len(original)
            assert np.isfinite(original).all()
            _,idx=np.unique(original,axis=0,return_index=True);idx=np.sort(idx);table=original[idx]
            raw=np.column_stack([idx+1,table[:,0],(table[:,1]-719529)*86400,table[:,3],table[:,4],table[:,5],table[:,7]])
            raw_path=dest/'raw'/cid/f'operation_{operation:02d}.npz';raw_path.parent.mkdir(parents=True,exist_ok=True)
            try:corrected,clock=repair_clock(raw)
            except ValueError as exc:
                corrected=raw;clock=[];issues.append(dict(spec,issue=str(exc)))
            np.savez_compressed(raw_path,original8=original,original_row_indices=idx,records=corrected)
            spec.update(cache=str(raw_path),cache_sha256=digest(raw_path),original_rows=n,staged_rows=len(raw),
                columns=columns,identical_rows_deduplicated=n-len(raw),clock_corrections=clock,**meta)
            raw_files.append(spec)
            ee,rr,bad=events(corrected,cid,protocol['nominal_capacity_Ah'],f'{meta["member"]}:operation_{operation:02d}',2.77)
            for e in ee:
                e.update(raw_cache=str(raw_path),raw_cache_sha256=spec['cache_sha256'],operation_index=operation)
            all_events.extend(ee);reject.update(rr);issues.extend(dict(spec,issue=b) for b in bad)
        all_events.sort(key=lambda e:e['charge_start_wallclock'])
        unique=[];seen={};duplicates=0;nominal=protocol['nominal_capacity_Ah']
        for e in all_events:
            key=e['charge_start_wallclock'];q=e['capacity_Ah']
            if key in seen:
                old=seen[key]
                if old['cutoff_wallclock']==e['cutoff_wallclock'] and np.array_equal(old['x'],e['x']) and abs(old['capacity_Ah']-q)<1e-8:
                    duplicates+=1;continue
                issues.append(dict(issue='conflicting_same_timestamp',member=e['member']));continue
            if unique and key<unique[-1]['cutoff_wallclock']:
                issues.append(dict(issue='overlapping_charge_discharge',member=e['member']))
            lo=e['sampled_current_integral_lower_Ah'];hi=e['sampled_current_integral_upper_Ah']
            if max(lo-q,q-hi,0)>=.001*nominal:
                issues.append(dict(issue='counter_integral_envelope_violation',member=e['member'],charge_start=key,capacity=q,lower=lo,upper=hi))
            fingerprint=hashlib.sha256(e['x'].tobytes()+np.asarray([q,key,e['cutoff_wallclock']],dtype=float).tobytes()).hexdigest()
            if fingerprint in fingerprints and fingerprints[fingerprint]!=cid:cross.append([cid,fingerprints[fingerprint]])
            fingerprints[fingerprint]=cid;seen[key]=e;unique.append(e)
        target=dest/'cells'/(cid+'.npz');target.parent.mkdir(exist_ok=True)
        if unique:
            np.savez_compressed(target,x=np.stack([e['x'] for e in unique]),capacity_Ah=np.asarray([e['capacity_Ah'] for e in unique]),
                acquisition_time=np.asarray([e['charge_start_wallclock'] for e in unique]),cycle_number=np.arange(1,len(unique)+1),
                cell_id=cid,domain=protocol['groups'][cid],nominal_capacity_Ah=nominal)
        record=dict(cell_id=cid,domain=protocol['groups'][cid],nominal_capacity_Ah=nominal,
            status='READY_FOR_PAIRING_VERIFICATION' if len(unique)>10 and not issues else 'INELIGIBLE_OR_INTEGRITY_REVIEW',
            measured_pairs=len(unique),query_points=max(0,len(unique)-10),integrity_issues=issues,
            omitted_files=omitted,rejections=dict(reject),raw_files=raw_files,duplicate_events=duplicates,
            cache=str(target) if unique else None,cache_sha256=digest(target) if unique else None,
            events=[{k:v for k,v in e.items() if k!='x'} for e in unique])
        write_json(dest/(cid+'_audit.json'),record);reports.append({k:v for k,v in record.items() if k not in ['events','raw_files']})
        print(cid,record['status'],'pairs',len(unique),'issues',len(issues),'omitted',len(omitted),flush=True)
    if cross:
        for r in reports:
            if any(r['cell_id'] in pair for pair in cross):r['status']='CROSS_CELL_DUPLICATE_REVIEW'
    write_json(dest/'summary.json',dict(status='EXTRACTED_NOT_APPROVED_FOR_PREDICTION',cells=reports,
        cross_cell_duplicate_pairs=cross,implementation_sha256=digest(dest/'implementation.json'),
        exposure_sha256=digest(OUT/'pl_exposure_audit.json'),model_predictions_generated=False))


if __name__=='__main__':main()
