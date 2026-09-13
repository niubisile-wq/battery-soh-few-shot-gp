"""Independent raw-row reconstruction gate before frozen-model prediction."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from study import HERE,PROTOCOL_PATH,digest,write_json,write_csv,preservation

OUT=HERE/'external'


def raw_clock_check(original,corrected,corrections):
    expected=np.asarray(original,dtype=float).copy()
    for c in corrections:
        i=c['after_row'];dt=expected[i+1,1]-expected[i,1];dw=expected[i+1,2]-expected[i,2]
        assert dt>0 and expected[i+1,0]>expected[i,0] and abs(dw-dt+3600)<2
        assert c['added_seconds']==3600
        expected[i+1:,2]+=3600
    np.testing.assert_array_equal(expected,corrected)
    assert (np.diff(corrected[:,2])>=0).all()


def window_from_indices(a,event):
    s=event['raw_charge_start_index'];stop=event['raw_charge_stop_index'];j=event['raw_window_start_index']-s
    v=a[s:stop,5];i=a[s:stop,4];t=a[s:stop,1]-a[s,1]
    assert v[j]<=3.9<v[j+1]
    b=next(k for k in range(j+1,len(v)-1) if v[k]<4.1<=v[k+1])
    assert b+2-j==event['window_raw_observations'] and b+2-j>=8
    # Check the declared first eligible traversal, without calling extract().
    for lower in range(j):
        if not v[lower]<=3.9<v[lower+1]:continue
        upper=next((k for k in range(lower+1,len(v)-1) if v[k]<4.1<=v[k+1]),None)
        assert upper is None or upper+2-lower<8
    left=t[j]+(3.9-v[j])/(v[j+1]-v[j])*(t[j+1]-t[j])
    right=t[b]+(4.1-v[b])/(v[b+1]-v[b])*(t[b+1]-t[b])
    grid=np.linspace(left,right,128)
    return np.vstack([np.interp(grid,t[j:b+2],v[j:b+2]),np.interp(grid,t[j:b+2],i[j:b+2]),grid-left]).astype('float32')


def check_pair(a,e,x,q,nominal,cutoff):
    s=e['raw_charge_start_index'];stop=e['raw_charge_stop_index'];d=e['raw_discharge_first_index'];end=e['raw_cutoff_index']
    assert 0<=s<=e['raw_window_start_index']<stop<=d<=end<len(a)
    assert (a[s:stop,4]>.01*nominal).all() and a[s:stop,5].max()>=4.18
    assert not (a[stop:end+1,4]>.01*nominal).any()
    assert d==stop+int(np.flatnonzero(a[stop:end+1,4]<-.05*nominal)[0])
    hits=np.flatnonzero((a[d:end+1,4]<-.05*nominal)&(a[d:end+1,5]<=cutoff))
    assert len(hits)==1 and d+int(hits[0])==end
    delta_t=np.diff(a[s:end+1,1]);delta_wall=np.diff(a[s:end+1,2])
    assert (delta_t>0).all() and not ((delta_wall-delta_t)>np.maximum(2.,2*delta_t)).any()
    assert e['charge_start_wallclock']==a[e['raw_window_start_index'],2]
    assert e['cutoff_wallclock']==a[end,2]
    xx=window_from_indices(a,e);xd=float(np.max(abs(xx-x)))
    assert xd<1e-4,('window mismatch',e['cell_id'],e['member'],xd)
    counter=a[d-1:end+1,6];assert (counter>=0).all()
    increments=[];resets=0
    for before,after in zip(counter[:-1],counter[1:]):
        if after<before-1e-8:increments.append(after);resets+=1
        else:increments.append(max(after-before,0.))
    reconstructed=math.fsum(increments)
    assert abs(reconstructed-q)<1e-10 and abs(reconstructed-e['capacity_Ah'])<1e-10
    assert resets==e['counter_resets']
    dt=np.diff(a[d-1:end+1,1]);current=np.maximum(-a[d-1:end+1,4],0)
    lo=float(np.dot(np.minimum(current[:-1],current[1:]),dt)/3600)
    hi=float(np.dot(np.maximum(current[:-1],current[1:]),dt)/3600)
    assert abs(lo-e['sampled_current_integral_lower_Ah'])<1e-9
    assert abs(hi-e['sampled_current_integral_upper_Ah'])<1e-9
    assert lo-.001*nominal<q<hi+.001*nominal
    return dict(cell_id=e['cell_id'],member=e['member'],window_rebuild_max_abs=xd,
        capacity_rebuild_abs_Ah=abs(reconstructed-q),capacity_Ah=q,integral_lower_Ah=lo,integral_upper_Ah=hi,
        measured_cutoff_V=float(a[end,5]),cutoff_rule_V=cutoff,clock=float(e['charge_start_wallclock']))


def main():
    preservation();raw_manifest=json.loads((OUT/'staging_summary.json').read_text())
    raw_lookup={(r['cell_id'],r['member']):r for r in raw_manifest['rows'] if r['status']=='STAGED'}
    cells=[];excluded=[];rows=[];input_hashes={};unique={};raw_count=0
    for directory in [OUT/'cohort',OUT/'pl_cohort']:
        summary=json.loads((directory/'summary.json').read_text());assert not summary['model_predictions_generated']
        assert not summary['cross_cell_duplicate_pairs']
        input_hashes[str(directory/'summary.json')]=digest(directory/'summary.json')
        input_hashes[str(directory/'implementation.json')]=digest(directory/'implementation.json')
        for brief in summary['cells']:
            cid=brief['cell_id'];audit_file=directory/(cid+'_audit.json');audit=json.loads(audit_file.read_text())
            input_hashes[str(audit_file)]=digest(audit_file)
            if brief['status']!='READY_FOR_PAIRING_VERIFICATION':
                excluded.append(dict(cell_id=cid,status=brief['status'],measured_pairs=brief['measured_pairs'],
                    integrity_issues=brief['integrity_issues'],omitted_files=len(brief['omitted_files']),rejections=brief['rejections']));continue
            assert not audit['integrity_issues'];assert digest(audit['cache'])==audit['cache_sha256']
            input_hashes[audit['cache']]=audit['cache_sha256']
            with np.load(audit['cache'],allow_pickle=False) as z:
                x=z['x'];q=z['capacity_Ah'];clock=z['acquisition_time'];cycle=z['cycle_number'];nominal=float(z['nominal_capacity_Ah'])
            assert x.shape==(len(q),3,128) and len(q)==len(audit['events'])==audit['measured_pairs']>10
            assert np.isfinite(x).all() and np.isfinite(q).all() and (q>0).all() and (np.diff(clock)>0).all()
            np.testing.assert_array_equal(cycle,np.arange(1,len(q)+1))
            for i,e in enumerate(audit['events']):
                assert clock[i]==e['charge_start_wallclock']
                if i:assert e['charge_start_wallclock']>=audit['events'][i-1]['cutoff_wallclock']
                h=hashlib.sha256(x[i].tobytes()+np.array([q[i],clock[i],e['cutoff_wallclock']],dtype=float).tobytes()).hexdigest()
                assert h not in unique,('duplicate observation',cid,unique.get(h));unique[h]=cid
            groups=defaultdict(list)
            for i,e in enumerate(audit['events']):groups[e['raw_cache']].append((i,e))
            for path,ee in groups.items():
                expected_sha=ee[0][1]['raw_cache_sha256'];assert digest(path)==expected_sha
                assert all(e['raw_cache_sha256']==expected_sha for _,e in ee);input_hashes[path]=expected_sha
                with np.load(path,allow_pickle=False) as z:
                    a=z['records']
                    if cid.startswith('PL'):
                        original8=z['original8'];idx=z['original_row_indices'];table=original8[idx]
                        original=np.column_stack([idx+1,table[:,0],(table[:,1]-719529)*86400,table[:,3],table[:,4],table[:,5],table[:,7]])
                        spec=next(s for s in audit['raw_files'] if s['cache']==path)
                        raw_clock_check(original,a,spec['clock_corrections'])
                if not cid.startswith('PL'):
                    spec=raw_lookup[cid,ee[0][1]['member']];assert digest(spec['cache'])==spec['cache_sha256']
                    with np.load(spec['cache'],allow_pickle=False) as z:original=z['records']
                    corrections=next((c['corrections'] for c in audit.get('clock_corrections',[]) if c['member']==spec['member']),[])
                    raw_clock_check(original,a,corrections);input_hashes[spec['cache']]=spec['cache_sha256']
                _,idx=np.unique(a,axis=0,return_index=True);a=a[np.sort(idx)]
                for i,e in ee:rows.append(check_pair(a,e,x[i],q[i],nominal,2.77 if cid.startswith('PL') else 2.72))
                raw_count+=1
            cells.append(dict(cell_id=cid,domain=brief['domain'],cache=audit['cache'],cache_sha256=audit['cache_sha256'],
                audit=str(audit_file),audit_sha256=digest(audit_file),measurements=len(q),support=10,query=len(q)-10,
                reference_capacity_Ah=float(np.float32(q[0])),nominal_capacity_Ah=nominal,
                raw_observed_capacity_min_Ah=float(q.min()),raw_observed_capacity_max_Ah=float(q.max()),
                incomplete_or_unverified_files_omitted=len(audit['omitted_files']),
                interpretation='First10 eligible measured charge/full-discharge pairs, not necessarily first10 physical cycles'))
            print('RAW_PAIRING_VERIFIED',cid,len(q),flush=True)
    expected=14;assert len(cells)+len(excluded)==expected
    for path in [PROTOCOL_PATH,OUT/'pl_protocol.json',OUT/'exposure_and_source_lock.json',OUT/'pl_source_lock.json',OUT/'pl_exposure_audit.json']:
        input_hashes[str(path)]=digest(path)
    assert len(rows)==sum(r['measurements'] for r in cells)
    write_csv(OUT/'raw_pair_verification.csv',rows)
    report=dict(status='APPROVED_FOR_FROZEN_PREDICTION',model_predictions_generated=False,candidate_cells=expected,
        eligible_cells=cells,excluded_cells=excluded,raw_files_reconstructed=raw_count,total_measurements=len(rows),
        total_query=sum(r['query'] for r in cells),all_pairs_rebuilt=True,
        max_window_rebuild_abs=max(r['window_rebuild_max_abs'] for r in rows),
        max_capacity_rebuild_abs_Ah=max(r['capacity_rebuild_abs_Ah'] for r in rows),
        input_hashes=input_hashes,verifier_sha256=digest(Path(__file__)),
        raw_pair_table_sha256=digest(OUT/'raw_pair_verification.csv'),
        scope='Previously unscored cells in an already used CALCE source; rate-dependent operational capacities; unequal protocol-group sizes; no cycle-level independence claim')
    write_json(OUT/'confirmation_data_gate.json',report)
    print('CONFIRMATION_DATA_APPROVED',len(cells),'cells',len(rows),'measurements',report['total_query'],'queries',flush=True)


if __name__=='__main__':main()
