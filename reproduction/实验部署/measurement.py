"""Measurement-only utilities. Clock corrections never change V/I/capacity."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from study import ROOT
sys.path.insert(0,str(ROOT/'开发基线选择依据'))
from partial_charge_protocol import Window,extract


def repair_clock(a):
    a=np.asarray(a,dtype=float).copy();corrections=[]
    for i in np.flatnonzero(np.diff(a[:,2])<0):
        dt=a[i+1,1]-a[i,1];dw=a[i+1,2]-a[i,2]
        if dt>0 and a[i+1,0]>a[i,0] and abs(dw-dt+3600)<2:
            a[i+1:,2]+=3600
            corrections.append(dict(after_row=int(i),raw_wallclock_delta=float(dw),instrument_delta=float(dt),
                                    added_seconds=3600,interpretation='One-hour clock rollback inferred from continuous instrument timer; not asserted to be a timezone identity'))
        else:raise ValueError(f'Unresolved backwards clock at row {i}')
    return a,corrections


def events(a,cid,nominal,member,cutoff):
    """Same measured-pair definition with explicit chemistry/protocol cutoff."""
    if not np.isfinite(a).all():return [],{'nonfinite_records':1},['nonfinite_records']
    _,ii=np.unique(a,axis=0,return_index=True);a=a[np.sort(ii)]
    if (np.diff(a[:,2])<0).any():return [],{},['backwards_clock']
    dt=np.diff(a[:,1]);dw=np.diff(a[:,2])
    breaks=np.flatnonzero((dt<=0)|((dw-dt)>np.maximum(2.,2*np.maximum(dt,0))))+1
    if len(breaks):
        result=[];reject=Counter();issues=[]
        for l,r in zip(np.r_[0,breaks],np.r_[breaks,len(a)]):
            ee,rr,xx=events(a[l:r],cid,nominal,member,cutoff)
            for e in ee:
                for key in ['raw_charge_start_index','raw_charge_stop_index','raw_window_start_index','raw_discharge_first_index','raw_cutoff_index']:
                    e[key]+=int(l)
            result.extend(ee);reject.update(rr);issues.extend(xx)
        reject['acquisition_boundaries']+=len(breaks)
        return result,dict(reject),issues
    signs=np.diff(np.r_[False,a[:,4]>.01*nominal,False].astype(int))
    runs=list(zip(np.flatnonzero(signs==1),np.flatnonzero(signs==-1)))
    out=[];reject=Counter()
    for i,(s,stop) in enumerate(runs):
        end=runs[i+1][0] if i+1<len(runs) else len(a)
        part=a[s:stop]
        if part[:,5].max()<4.18:reject['incomplete_charge']+=1;continue
        try:x,meta=extract(dict(voltage_in_V=part[:,5],current_in_A=part[:,4],time_in_s=part[:,1]-part[0,1],cycle_number=i),Window(3.9,4.1))
        except ValueError as exc:reject[str(exc)]+=1;continue
        under=np.flatnonzero(a[stop:end,4]<-.05*nominal)+stop
        if not len(under):reject['no_following_discharge']+=1;continue
        hits=under[a[under,5]<=cutoff]
        if not len(hits):reject['no_measured_full_reference_cutoff']+=1;continue
        first=int(under[0]);last=int(hits[0]);seg=a[first-1:last+1]
        q=seg[:,6];dq=np.diff(q)
        capacity=float(np.where(dq < -1e-8,q[1:],np.maximum(dq,0.)).sum())
        if capacity<=0:reject['nonpositive_increment']+=1;continue
        tt=np.diff(seg[:,1]);i0=np.maximum(-seg[:-1,4],0);i1=np.maximum(-seg[1:,4],0)
        lower=float(np.sum(np.minimum(i0,i1)*tt)/3600);upper=float(np.sum(np.maximum(i0,i1)*tt)/3600)
        out.append(dict(cell_id=cid,member=member,charge_start_wallclock=float(a[s+meta['sample_start'],2]),
            cutoff_wallclock=float(a[last,2]),raw_charge_start_index=int(s),raw_charge_stop_index=int(stop),
            raw_window_start_index=int(s+meta['sample_start']),raw_discharge_first_index=first,raw_cutoff_index=last,
            counter_resets=int((dq < -1e-8).sum()),capacity_Ah=capacity,
            current_integral_Ah=(lower+upper)/2,counter_integral_difference_Ah=abs(capacity-(lower+upper)/2),
            sampled_current_integral_lower_Ah=lower,sampled_current_integral_upper_Ah=upper,
            discharge_cutoff_V=float(a[last,5]),reference_cutoff_V=cutoff,
            cycle_index_at_charge=float(a[s,3]),cycle_index_at_cutoff=float(a[last,3]),
            window_raw_observations=int(meta['sample_stop_exclusive']-meta['sample_start']),x=x))
    return out,dict(reject),[]
