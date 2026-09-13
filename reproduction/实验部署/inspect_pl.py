"""Decode public PL MATLAB tables into typed Python objects, no predictions."""
from pathlib import Path
import json
import shutil
import sys
import zipfile
from study import HERE,ROOT,digest,write_json

sys.path.insert(0,str(HERE/'vendor'))
from matio import load_from_mat


def load_cell(cid):
    base=HERE/'external';lock=json.loads((base/'pl_source_lock.json').read_text())
    assert lock['protocol_sha256']==digest(base/'pl_protocol.json')
    archive=ROOT/'数据集/CALCE/PL'/('SOC_0-100_HalfC.zip' if cid in ['PL11','PL13'] else 'SOC_0-100_2C.zip')
    assert digest(archive)==lock['archive_sha256'][str(archive)]
    target=base/'pl_inputs'/(cid+'.mat');target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        member=next(n for n in z.namelist() if n.endswith(cid+'.mat'))
        if not target.exists():
            with z.open(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
        with z.open(member) as src:
            import hashlib
            h=hashlib.sha256()
            for b in iter(lambda:src.read(1048576),b''):h.update(b)
        assert digest(target)==h.hexdigest()
    values=load_from_mat(str(target),raw_data=True)
    return values[cid],dict(cell_id=cid,archive=str(archive),member=member,mat_sha256=digest(target))


def old_table(table):
    """Named R2014 table columns; do not infer order from anonymous numbers."""
    import numpy as np
    assert table.classname=='table'
    props=table.properties
    columns=[str(np.asarray(v).item()) for v in props['varnames'].ravel()]
    expected=['Time_sec','Date_Time','Step','Cycle','Current_Amp','Voltage_Volt','Charge_Ah','Discharge_Ah']
    assert columns==expected,columns
    count=int(props['nrows'].item())
    assert int(props['nvars'].item())==8
    arrays=[np.asarray(v,dtype=float).reshape(-1) for v in props['data'].ravel()]
    assert all(len(v)==count for v in arrays)
    return np.column_stack(arrays),columns


if __name__=='__main__':
    v,meta=load_cell('PL11')
    print(meta,'type',type(v),'shape',getattr(v,'shape',None))
    for row in v[:3]:
        print('operation',repr(row[0]),'type',type(row[2]))
        t=row[2]
        if hasattr(t,'columns'):print('columns',t.columns.tolist(),'shape',t.shape,'head',t.iloc[:2,:6].to_dict())
        else:print(repr(t)[:600])
