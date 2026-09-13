"""Prove quality correction did not change support, retained labels or predictions."""
import json
from pathlib import Path
import numpy as np
from evaluate import sa,HERE


def main():
    root=sa.ROOT/'模块研发/results';old=root/'calce_cohort_extraction_v2';new=root/'calce_cohort_extraction_v3'
    previous=root/'calce_frozen_confirmation_v1';current=root/'calce_frozen_confirmation_v2'
    old_request=json.loads((previous/'request.json').read_text());new_request=json.loads((current/'request.json').read_text())
    assert old_request['candidate_sha256']==new_request['candidate_sha256']
    assert old_request['source_lock_sha256']==new_request['source_lock_sha256']
    report=json.loads((current/'summary.json').read_text());removed=[];count=0;maximum=0.
    for cid in ['CS2_3','CS2_9','CX2_8','CX2_32']:
        a=json.loads((old/(cid+'_audit.json')).read_text());b=json.loads((new/(cid+'_audit.json')).read_text())
        old_times=[r['charge_start_wallclock'] for r in a['events']];new_times=[r['charge_start_wallclock'] for r in b['events']]
        assert old_times[:10]==new_times[:10]
        old_lookup={t:j for j,t in enumerate(old_times)};assert set(new_times)<=set(old_times)
        with np.load(old/'cells'/(cid+'.npz')) as z:ox=z['x'];oy=z['capacity_Ah']
        with np.load(new/'cells'/(cid+'.npz')) as z:nx=z['x'];ny=z['capacity_Ah']
        keep=np.array([old_lookup[t] for t in new_times]);np.testing.assert_array_equal(oy[keep],ny);np.testing.assert_array_equal(ox[keep],nx)
        removed.extend(dict(cell_id=cid,member=e['member'],charge_start_wallclock=e['charge_start_wallclock'])
                       for e in a['events'] if e['charge_start_wallclock'] not in set(new_times))
        qkeep=keep[10:]-10
        for row in report['rows']:
            if row['cell_id']!=cid:continue
            with np.load(previous/row['prediction_file']) as z:op=z['pred'];ot=z['y']
            with np.load(current/row['prediction_file']) as z:npred=z['pred'];nt=z['y']
            np.testing.assert_array_equal(ot[qkeep],nt)
            delta=float(np.max(abs(op[qkeep]-npred)));assert delta<1e-8
            maximum=max(maximum,delta);count+=1
    result=dict(status='PASS',unchanged_support_cells=4,removed_measurements=len(removed),removed=removed,
        verified_cell_fold_group_rows=count,max_retained_prediction_change=maximum,
        source_model_locks_unchanged=True,retained_signals_and_labels_unchanged=True,
        code_sha256=sa.digest(Path(__file__)),
        disclosure='Correction made after preliminary scoring, based on raw wall-clock/timer consistency, applied identically to every model. Not a new independent test and no M3/model retuning.')
    assert len(removed)==4 and count==144
    sa.write_json(current/'correction_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
