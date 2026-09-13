"""Signal-defined charge/full-discharge pairing; no model evaluation.

Arbin Cycle_Index may be constant across real cycles. Capacity counters may
accumulate across cycles. Neither field is assumed to reset per cycle.
"""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from evaluate import sa, HERE
sys.path.insert(0,str(sa.ROOT/'开发基线选择依据'))
from partial_charge_protocol import Window,extract


def positive_runs(current, threshold):
    b=np.diff(np.r_[False,np.asarray(current)>threshold,False].astype(int))
    return list(zip(np.flatnonzero(b==1),np.flatnonzero(b==-1)))


def counter_increment(counter):
    q=np.asarray(counter,dtype=float)
    if not np.isfinite(q).all() or (q<0).any():raise ValueError('Invalid counter')
    d=np.diff(q);reset=d < -1e-8
    # On an instrument reset, the newly recorded value is the new increment.
    return float(np.sum(np.where(reset,q[1:],np.maximum(d,0)))),int(reset.sum())


def events_from_book(a,cid,nominal,member):
    events=[];reject=Counter();issues=[]
    if not np.isfinite(a).all():return [],{'nonfinite_workbook':1},['nonfinite_workbook']
    _,index=np.unique(a,axis=0,return_index=True)
    duplicates=len(a)-len(index);a=a[np.sort(index)]
    if (np.diff(a[:,2])<0).any():
        return [],{'nonmonotone_workbook':1},['nonmonotone_workbook']
    timer_delta=np.diff(a[:,1]);wall_delta=np.diff(a[:,2])
    unexplained_gap=(wall_delta-timer_delta)>np.maximum(2.,2*np.maximum(timer_delta,0))
    breaks=np.flatnonzero((timer_delta<=0)|unexplained_gap)+1
    if len(breaks):
        # Instrument timer restart with forward wall clock is a new session,
        # not permission to integrate across the downtime or discard the cell.
        chunks=np.r_[0,breaks,len(a)]
        for left,right in zip(chunks[:-1],chunks[1:]):
            ee,rr,ii=events_from_book(a[left:right],cid,nominal,member)
            for event in ee:
                for key in ['raw_charge_start_index','raw_charge_stop_index','raw_window_start_index',
                            'raw_discharge_first_index','raw_cutoff_index']:event[key]+=int(left)
            events.extend(ee);reject.update(rr);issues.extend(ii)
        reject['instrument_timer_session_boundaries']+=len(breaks)
        reject['wallclock_pause_boundaries']+=int(unexplained_gap.sum())
        reject['identical_raw_rows_deduplicated']+=duplicates
        return events,dict(reject),issues
    runs=positive_runs(a[:,4],.01*nominal)
    for k,(s,e) in enumerate(runs):
        end=runs[k+1][0] if k+1<len(runs) else len(a)
        part=a[s:e]
        if part[:,5].max()<4.18:
            reject['charge_did_not_reach_4.18V']+=1;continue
        raw=dict(voltage_in_V=part[:,5],current_in_A=part[:,4],time_in_s=part[:,1]-part[0,1],cycle_number=k)
        try:x,meta=extract(raw,Window(3.9,4.1))
        except ValueError as exc:reject[str(exc)]+=1;continue
        under=np.flatnonzero(a[e:end,4]<-.05*nominal)+e
        if not len(under):reject['no_following_discharge_under_load']+=1;continue
        hits=under[a[under,5]<=2.72]
        if not len(hits):reject['no_measured_2.7V_reference_cutoff']+=1;continue
        last=int(hits[0]);first=int(under[0])
        # Include the previous non-discharging observation for counter baseline.
        q,resets=counter_increment(a[first-1:last+1,6])
        if q<=0:reject['nonpositive_measured_increment']+=1;continue
        seg=a[first-1:last+1]
        integral=float(np.sum(.5*(np.maximum(-seg[:-1,4],0)+np.maximum(-seg[1:,4],0))*np.diff(seg[:,1]))/3600)
        left_current=np.maximum(-seg[:-1,4],0);right_current=np.maximum(-seg[1:,4],0)
        lower=float(np.sum(np.minimum(left_current,right_current)*np.diff(seg[:,1]))/3600)
        upper=float(np.sum(np.maximum(left_current,right_current)*np.diff(seg[:,1]))/3600)
        start_index=s+meta['sample_start']
        events.append(dict(cell_id=cid,member=member,charge_start_wallclock=float(a[start_index,2]),
            cutoff_wallclock=float(a[last,2]),raw_charge_start_index=int(s),raw_charge_stop_index=int(e),
            raw_window_start_index=int(start_index),raw_discharge_first_index=first,raw_cutoff_index=last,
            counter_resets=resets,capacity_Ah=q,current_integral_Ah=integral,
            counter_integral_difference_Ah=abs(q-integral),discharge_cutoff_V=float(a[last,5]),
            sampled_current_integral_lower_Ah=lower,sampled_current_integral_upper_Ah=upper,
            cycle_index_at_charge=float(a[s,3]),cycle_index_at_cutoff=float(a[last,3]),
            window_raw_observations=int(meta['sample_stop_exclusive']-meta['sample_start']),x=x))
    reject['identical_raw_rows_deduplicated']+=duplicates
    return events,dict(reject),issues


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    staging=sa.ROOT/'模块研发/results/calce_raw_staging_v1'
    request=json.loads((staging/'request.json').read_text());assert request['status']=='COMPLETE'
    manifest=json.loads((staging/'summary.json').read_text())
    protocol=json.loads((HERE/'calce_cohort_protocol.json').read_text())
    assert request['protocol_sha256']==sa.digest(HERE/'calce_cohort_protocol.json')
    implementation=dict(status='DEFINED_BEFORE_COHORT_SCORING',positive_charge_threshold_fraction_nominal=.01,
        discharge_threshold_fraction_nominal=.05,charge_complete_voltage=4.18,reference_cutoff_tolerance_V=.02,
        capacity='Observed counter increments through first under-load sample at or below 2.72 V; counter resets explicitly accumulated',
        counter_crosscheck='Compare with current-time quadrature; discrepancies are diagnostic, not prediction-based filters',
        cycle_id='Signal-defined charge events; file-local Cycle_Index retained for audit only',
        chronology='Split timer resets or unexplained wall-clock gaps: wall_delta - timer_delta > max(2 seconds, 2*timer_delta). This catches unobserved downtime, not model errors. Never join charge/discharge across these boundaries. No merging across files; incomplete edges excluded; identical events deduplicated, conflicting overlaps block approval.',
        limits='Recorded reference capacity at a fixed threshold; current-rate dependence remains. No label interpolation or model scores.')
    sa.write_json(out/'implementation.json',implementation)
    reports=[]
    for cid in protocol['primary_candidates']:
        all_events=[];rejections=Counter();issues=[]
        for f in [r for r in manifest['files'] if r['cell_id']==cid]:
            assert sa.digest(f['cache'])==f['cache_sha256']
            with np.load(f['cache']) as z:a=z['records']
            events,rej,bad=events_from_book(a,cid,protocol['nominal_capacity_Ah'][cid],f['member'])
            all_events.extend(events);rejections.update(rej);issues.extend(dict(member=f['member'],issue=b) for b in bad)
        all_events.sort(key=lambda r:(r['charge_start_wallclock'],r['member']))
        unique=[];seen={};duplicates=0
        for e in all_events:
            key=e['charge_start_wallclock']
            if key in seen:
                old=seen[key]
                if (old['cutoff_wallclock']==e['cutoff_wallclock'] and np.array_equal(old['x'],e['x'])
                        and abs(old['capacity_Ah']-e['capacity_Ah'])<1e-8):duplicates+=1;continue
                issues.append(dict(issue='conflicting_same_charge_timestamp',members=[old['member'],e['member']]))
                continue
            if unique and e['charge_start_wallclock']<unique[-1]['cutoff_wallclock']:
                issues.append(dict(issue='overlapping_charge_discharge_intervals',members=[unique[-1]['member'],e['member']]))
            seen[key]=e;unique.append(e)
        target=out/'cells'/(cid+'.npz');target.parent.mkdir(exist_ok=True)
        if unique:
            np.savez_compressed(target,x=np.stack([e['x'] for e in unique]),
                capacity_Ah=np.array([e['capacity_Ah'] for e in unique]),
                acquisition_time=np.array([e['charge_start_wallclock'] for e in unique]),
                cycle_number=np.arange(1,len(unique)+1),cell_id=cid,
                nominal_capacity_Ah=protocol['nominal_capacity_Ah'][cid],domain=protocol['protocol_groups'][cid])
        cross=[e['counter_integral_difference_Ah'] for e in unique]
        row=dict(cell_id=cid,status='PENDING_LABEL_ALIGNMENT_VERIFICATION',candidate_measured_cycles=len(unique),
            first_K_available=len(unique)>10,rejections=dict(rejections),identical_events_deduplicated=duplicates,
            integrity_issues=issues,cache=str(target) if unique else None,
            cache_sha256=sa.digest(target) if unique else None,
            max_counter_integral_difference_Ah=max(cross) if cross else None,
            median_counter_integral_difference_Ah=float(np.median(cross)) if cross else None,
            events=[{k:v for k,v in e.items() if k!='x'} for e in unique])
        sa.write_json(out/(cid+'_audit.json'),row);reports.append({k:v for k,v in row.items() if k!='events'})
        print(cid,'candidate cycles',len(unique),'integrity issues',len(issues),flush=True)
    sa.write_json(out/'summary.json',dict(status='EXTRACTION_AUDITED_NOT_APPROVED_FOR_SCORING',cells=reports,
        code_sha256=sa.digest(Path(__file__)),raw_manifest_sha256=sa.digest(staging/'summary.json'),
        protocol_sha256=sa.digest(HERE/'calce_cohort_protocol.json'),model_predictions_generated=False))


if __name__=='__main__':main()
