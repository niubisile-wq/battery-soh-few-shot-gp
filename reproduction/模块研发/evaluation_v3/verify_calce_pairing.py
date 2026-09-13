"""Reconstruct every admitted signal/label pair from raw numeric staging."""
import json
import argparse
from pathlib import Path
import numpy as np
from evaluate import sa,HERE
from build_calce_cohort import Window,extract


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',default='calce_cohort_extraction_v3');args=ap.parse_args()
    root=sa.ROOT/'模块研发/results';cohort=root/args.cohort
    raw=json.loads((root/'calce_raw_staging_v1/summary.json').read_text())
    protocol=json.loads((HERE/'calce_cohort_protocol.json').read_text())
    summaries=json.loads((cohort/'summary.json').read_text())
    assert summaries['code_sha256']==sa.digest(HERE/'build_calce_cohort.py')
    count=0;max_q_error=max_x_error=max_envelope_violation=0.;checks=[]
    for cid in protocol['primary_candidates']:
        record=json.loads((cohort/(cid+'_audit.json')).read_text())
        assert not record['integrity_issues'] and record['first_K_available']
        assert sa.digest(record['cache'])==record['cache_sha256']
        with np.load(record['cache']) as z:
            x=z['x'];q=z['capacity_Ah'];times=z['acquisition_time']
        assert len(q)==len(record['events']) and np.isfinite(q).all() and (q>0).all()
        assert (np.diff(times)>0).all()
        books={}
        for row in raw['files']:
            if row['cell_id']!=cid:continue
            assert sa.digest(row['cache'])==row['cache_sha256']
            with np.load(row['cache']) as z:a=z['records']
            _,ii=np.unique(a,axis=0,return_index=True);books[row['member']]=a[np.sort(ii)]
        for j,e in enumerate(record['events']):
            a=books[e['member']];s=e['raw_charge_start_index'];stop=e['raw_charge_stop_index']
            first=e['raw_discharge_first_index'];end=e['raw_cutoff_index']
            nominal=protocol['nominal_capacity_Ah'][cid]
            assert s<stop<=first<=end<len(a)
            assert (a[s:stop,4]>.01*nominal).all() and a[s:stop,5].max()>=4.18
            assert (a[stop:end+1,4]<=.01*nominal).all()
            assert a[first,4]<-.05*nominal and a[end,4]<-.05*nominal and a[end,5]<=2.72
            previous=a[first:end];assert not ((previous[:,4]<-.05*nominal)&(previous[:,5]<=2.72)).any()
            assert (np.diff(a[s:end+1,1])>0).all() and (np.diff(a[s:end+1,2])>=0).all()
            dt=np.diff(a[s:end+1,1]);dw=np.diff(a[s:end+1,2])
            assert not ((dw-dt)>np.maximum(2.,2*dt)).any()
            part=a[s:stop]
            rebuilt,meta=extract(dict(voltage_in_V=part[:,5],current_in_A=part[:,4],
                time_in_s=part[:,1]-part[0,1],cycle_number=j),Window(3.9,4.1))
            q_rebuilt=0.
            for before,after in zip(a[first-1:end,6],a[first:end+1,6]):
                q_rebuilt+=after if after<before-1e-8 else max(after-before,0.)
            dx=float(np.max(abs(rebuilt-x[j])));dq=abs(q_rebuilt-q[j])
            max_x_error=max(max_x_error,dx);max_q_error=max(max_q_error,dq)
            assert dx<1e-10 and dq<1e-10
            assert a[s+meta['sample_start'],2]==times[j]==e['charge_start_wallclock']
            lower=e['sampled_current_integral_lower_Ah'];upper=e['sampled_current_integral_upper_Ah']
            violation=max(lower-q[j],q[j]-upper,0.)
            max_envelope_violation=max(max_envelope_violation,violation)
            # Unit/measurement consistency check, not SOH error filtering.
            assert violation<.001*nominal
            count+=1
        checks.append(dict(cell_id=cid,eligible_measured_cycles=len(q),support_points=10,query_points=len(q)-10,
                           cache_sha256=record['cache_sha256']))
    lock=json.loads((root/'calce_cohort_schema_audit_v1/audit.json').read_text())
    assert lock['protocol_sha256']==sa.digest(HERE/'calce_cohort_protocol.json')
    assert all(sa.digest(sa.ROOT/r['path'])==r['sha256'] for r in lock['source_artifacts_locked'])
    report=dict(status='PASS_FOR_FROZEN_CELL_COHORT_CONFIRMATION',verified_pairs=count,cells=checks,
        max_signal_reconstruction_error=max_x_error,max_capacity_reconstruction_error_Ah=max_q_error,
        max_counter_current_envelope_violation_Ah=max_envelope_violation,
        protocol_sha256=sa.digest(HERE/'calce_cohort_protocol.json'),code_sha256=sa.digest(Path(__file__)),
        extraction_summary_sha256=sa.digest(cohort/'summary.json'),
        limits=['Same CALCE laboratory already appeared in earlier baselines; newly audited cell cohort, not new laboratory.',
                'Evidence of no prior cell-ID references is limited to audited workspace.',
                'Capacity at measured 2.7V reference depends on discharge rate; small four-cell cohort.',
                'Counter/current agreement tolerance 0.1% nominal is an engineering check set during raw-data diagnosis, not an efficacy preregistration.',
                'First K eligible observed measurements, not proof of first K physical cycles from manufacture.',
                'Incomplete acquisition edges excluded; no synthetic/interpolated capacity labels.'])
    sa.write_json(cohort/'verification.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
