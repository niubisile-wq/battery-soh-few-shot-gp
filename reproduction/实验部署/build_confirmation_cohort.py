"""Admit only measured, chronologically unambiguous full-reference observations."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from study import HERE, ROOT, PROTOCOL, PROTOCOL_PATH, digest, write_json

sys.path.insert(0,str(ROOT/'模块研发/evaluation_v3'))
from build_calce_cohort import events_from_book
from measurement import repair_clock

OUT=HERE/'external'
GROUPS={
    'CS2_5':'CS2_low_SOC_RPT_0.22A','CS2_6':'CS2_low_SOC_RPT_0.22A',
    'CS2_7':'CS2_variable_cutoff_full_reference','CS2_8':'CS2_0.5C','CS2_21':'CS2_0.5C',
    'CS2_24':'CS2_high_SOC_RPT_0.22A','CS2_25':'CS2_high_SOC_RPT_0.22A',
    'CX2_3':'CX2_alternating_pulse_full_reference','CX2_4':'CX2_temperature_1C','CX2_31':'CX2_0.5C'}


def main():
    raw=json.loads((OUT/'staging_summary.json').read_text())
    assert raw['status']=='RAW_STAGING_DONE_NOT_ELIGIBILITY_APPROVAL'
    lock=json.loads((OUT/'exposure_and_source_lock.json').read_text())
    assert lock['status']=='NO_MATCH_IN_AUDITED_RESULT_HISTORY'
    assert lock['protocol_sha256']==digest(PROTOCOL_PATH)
    dest=OUT/'cohort';dest.mkdir(exist_ok=True)
    rules=dict(created_before_model_prediction=True,all_candidate_ids=PROTOCOL['external_confirmation']['candidate_ids'],
        arbin_rule='Reuse chronological signal-defined full-charge/full-discharge extraction, no joining acquisition files',
        txt_rule='TXT time/capacity units and global chronology are not sufficiently documented by numeric headers. Archive raw values but do not assign SOH labels from an assumed scale. Such records do not qualify for primary confirmation.',
        reference='Full charge>=4.18 V followed by first measured under-load cutoff<=2.72 V',
        counter_crosscheck='Counter capacity must lie inside lower/upper sampled-current integral envelope plus 0.1% nominal tolerance. A violation blocks the cell for review, not silently discarded to improve accuracy.',
        clock_rule='A backwards wall-clock jump is repaired only if Data_Point and instrument time advance and wall_delta-instrument_delta is -3600 seconds within2 seconds. Preserve raw values and record a +3600 second logical-clock adjustment. No V/I/capacity changes.',
        missing_records='Unqualified records omitted from eligibility, with every file recorded. First10 refers only to eligible real measurements, not first10 physical ageing cycles.',
        groups=GROUPS,external_protocol_sha256=digest(PROTOCOL_PATH),
        extractor_sha256=digest(ROOT/'模块研发/evaluation_v3/build_calce_cohort.py'),code_sha256=digest(Path(__file__)))
    write_json(dest/'implementation.json',rules)
    report=[];fingerprints=defaultdict(set)
    for cid in PROTOCOL['external_confirmation']['candidate_ids']:
        nominal=1.1 if cid.startswith('CS2') else 1.35
        specs=[r for r in raw['rows'] if r['cell_id']==cid]
        events=[];reject=Counter();issues=[];omitted=[];clock_records=[]
        for spec in specs:
            if spec['status']!='STAGED':
                omitted.append(dict(member=spec['member'],reason=spec['status'],details=spec.get('error')));continue
            if spec['schema']!='arbin':
                omitted.append(dict(member=spec['member'],reason='UNVERIFIED_TXT_UNITS_AND_ABSOLUTE_CHRONOLOGY'));continue
            assert digest(spec['cache'])==spec['cache_sha256']
            with np.load(spec['cache'],allow_pickle=False) as z:a=z['records']
            try:a,clock=repair_clock(a)
            except ValueError as exc:
                issues.append(dict(member=spec['member'],issue=str(exc)));continue
            if clock:
                clock_records.append(dict(member=spec['member'],corrections=clock))
                corrected=dest/'corrected_raw'/cid/(Path(spec['cache']).name)
                corrected.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(corrected,records=a)
                spec=dict(spec,cache=str(corrected),cache_sha256=digest(corrected))
            ee,rr,ii=events_from_book(a,cid,nominal,spec['member'])
            for e in ee:e['raw_cache']=spec['cache'];e['raw_cache_sha256']=spec['cache_sha256']
            events.extend(ee);reject.update(rr)
            issues.extend(dict(member=spec['member'],issue=i) for i in ii)
        events.sort(key=lambda e:(e['charge_start_wallclock'],e['member']))
        unique=[];seen={};duplicates=0
        for e in events:
            key=e['charge_start_wallclock']
            if key in seen:
                old=seen[key]
                if old['cutoff_wallclock']==e['cutoff_wallclock'] and np.array_equal(old['x'],e['x']) and abs(old['capacity_Ah']-e['capacity_Ah'])<1e-8:
                    duplicates+=1;continue
                issues.append(dict(issue='conflicting_same_charge_timestamp',members=[old['member'],e['member']]))
                continue
            if unique and e['charge_start_wallclock']<unique[-1]['cutoff_wallclock']:
                issues.append(dict(issue='overlapping_charge_discharge_intervals',members=[unique[-1]['member'],e['member']]))
            q=e['capacity_Ah'];lo=e['sampled_current_integral_lower_Ah'];hi=e['sampled_current_integral_upper_Ah']
            violation=max(lo-q,q-hi,0.)
            if violation>=.001*nominal:
                issues.append(dict(issue='counter_integral_envelope_violation',member=e['member'],charge_start=key,violation_Ah=violation))
            fingerprint=hashlib.sha256(e['x'].tobytes()+np.asarray([q,key,e['cutoff_wallclock']],dtype=float).tobytes()).hexdigest()
            fingerprints[fingerprint].add(cid)
            seen[key]=e;unique.append(e)
        path=dest/'cells'/(cid+'.npz')
        if unique:
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,x=np.stack([e['x'] for e in unique]),capacity_Ah=np.asarray([e['capacity_Ah'] for e in unique]),
                acquisition_time=np.asarray([e['charge_start_wallclock'] for e in unique]),
                cycle_number=np.arange(1,len(unique)+1),cell_id=cid,domain=GROUPS[cid],nominal_capacity_Ah=nominal)
        status='READY_FOR_PAIRING_VERIFICATION' if len(unique)>10 and not issues else 'INELIGIBLE_OR_INTEGRITY_REVIEW'
        record=dict(cell_id=cid,domain=GROUPS[cid],status=status,measured_pairs=len(unique),query_points=max(0,len(unique)-10),
            nominal_capacity_Ah=nominal,integrity_issues=issues,omitted_files=omitted,rejections=dict(reject),
            duplicate_events=duplicates,cache=str(path) if unique else None,cache_sha256=digest(path) if unique else None,
            clock_corrections=clock_records,events=[{k:v for k,v in e.items() if k!='x'} for e in unique])
        write_json(dest/(cid+'_audit.json'),record)
        report.append({k:v for k,v in record.items() if k!='events'})
        print(cid,status,'pairs',len(unique),'issues',len(issues),'omitted_files',len(omitted),flush=True)
    cross=[sorted(v) for v in fingerprints.values() if len(v)>1]
    if cross:
        for r in report:
            if any(r['cell_id'] in x for x in cross):r['status']='CROSS_CELL_DUPLICATE_REVIEW'
    write_json(dest/'summary.json',dict(status='EXTRACTED_NOT_APPROVED_FOR_PREDICTION',cells=report,
        cross_cell_duplicate_pairs=cross,source_staging_sha256=digest(OUT/'staging_summary.json'),
        implementation_sha256=digest(dest/'implementation.json'),model_predictions_generated=False))


if __name__=='__main__':main()
