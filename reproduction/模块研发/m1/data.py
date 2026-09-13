"""Isolated first-module development data; preserves the champion experiment."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
K = PROTOCOL['K']
CACHE = ROOT / '实验部署/m1_development_cache_v1'
OUT = ROOT / '模块研发/results/m1_v1'


def natural(s):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', str(s))]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@dataclass
class Cell:
    id: str
    dataset: str
    domain: str
    x: np.ndarray
    y: np.ndarray
    cycle: np.ndarray
    reference_capacity: float
    nominal_capacity: float
    path: str = ''


def read_cell(path, dataset):
    with np.load(path, allow_pickle=False) as z:
        cid = str(z['cell_id'])
        x = z['x'].astype('float32')
        q = z['capacity_Ah'].astype('float32')
        cycle = z['cycle_number'].astype('float32')
        nominal = float(z['nominal_capacity_Ah'])
        if dataset == 'XJTU':
            domain = cid.split('XJTU_')[-1].rsplit('_battery', 1)[0]
        else:
            domain = str(z['domain'])
    if (len(q) <= K or x.shape != (len(q), 3, 128)
            or any(not np.isfinite(a).all() for a in [x, q, cycle])
            or (q <= 0).any() or (np.diff(cycle) <= 0).any()):
        raise ValueError(f'Invalid eligible series: {path}')
    return Cell(cid, dataset, domain, x, q/q[0], cycle, float(q[0]), nominal,
                str(Path(path).relative_to(ROOT)))


def load_cells(dataset):
    if dataset == 'XJTU':
        paths = sorted((ROOT / '实验部署/source_window_cache_v2/offset_03_01').glob('*/*.npz'))
    elif dataset in ['MATR', 'Tongji']:
        manifest = json.loads((CACHE/dataset/'manifest.json').read_text())
        if manifest['status'] != 'COMPLETE':
            raise RuntimeError(f'{dataset} cache unfinished')
        paths = [CACHE/dataset/r['path'] for r in manifest['cells'] if r['eligible']]
    else:
        raise ValueError('Only declared development datasets are permitted')
    cells = [read_cell(p, dataset) for p in paths]
    assert len(cells) == len({c.id for c in cells})
    return cells


def cap_training_cells(cells, maximum=None):
    maximum = PROTOCOL['max_source_cells'] if maximum is None else maximum
    if len(cells) <= maximum:
        return sorted(cells, key=lambda c: natural(c.id))
    groups = defaultdict(list)
    for c in cells:
        groups[(c.dataset,c.domain)].append(c)
    groups = {g:sorted(v,key=lambda c:natural(c.id)) for g,v in sorted(groups.items())}
    quotas = dict.fromkeys(groups,0)
    while sum(quotas.values()) < maximum:
        for g in groups:
            if quotas[g] < len(groups[g]) and sum(quotas.values()) < maximum:
                quotas[g] += 1
    selected = []
    for g, group in groups.items():
        ix = np.linspace(0,len(group)-1,quotas[g]).astype(int)
        selected.extend(group[i] for i in ix)
    return sorted(selected, key=lambda c:natural(c.id))


def split(cells, fold):
    if fold.startswith('LODO:'):
        held = fold.split(':',1)[1]
        target = [c for c in cells if c.dataset == held]
        pool = [c for c in cells if c.dataset != held]
    else:
        dataset,domain = fold.split(':',1)
        target = [c for c in cells if c.dataset == dataset and c.domain == domain]
        pool = [c for c in cells if c.dataset == dataset and c.domain != domain]
    groups=defaultdict(list)
    for c in pool:
        groups[(c.dataset,c.domain)].append(c)
    train,val=[],[]
    for _,g in sorted(groups.items()):
        g=sorted(g,key=lambda c:natural(c.id))
        if len(g)<2:
            raise ValueError('Need separate training and validation cells in every source group')
        train.extend(g[:-1]);val.append(g[-1])
    train=cap_training_cells(train)
    ids=[{c.id for c in g} for g in [train,val,target]]
    if not all(ids) or any(ids[i]&ids[j] for i,j in [(0,1),(0,2),(1,2)]):
        raise ValueError('Invalid cell split')
    return train,val,target


def source_indices(cells):
    cells=sorted(cells,key=lambda c:natural(c.id))
    budget=PROTOCOL['source_label_budget'];result={}
    for i,c in enumerate(cells):
        count=min(budget//len(cells)+int(i<budget%len(cells)),len(c.y))
        if count<K:
            raise ValueError('Budget cannot include the required support labels')
        idx=np.arange(K)
        if count>K:
            idx=np.r_[idx,np.linspace(K,len(c.y)-1,count-K).astype(int)]
        result[c.id]=idx
    return result


def provenance(train,val,target,indices):
    return {
        'train_cells':[c.id for c in train], 'validation_cells':[c.id for c in val],
        'development_target_cells':[c.id for c in target],
        'source_keys':[[c.id,float(c.cycle[j])] for c in train for j in indices[c.id]],
        'support_cycles':{c.id:c.cycle[:K].tolist() for c in val+target},
        'protocol_sha256':digest(HERE/'protocol.json'),
        'code_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},
        'cache_sha256':{c.path:digest(ROOT/c.path) for c in train+val+target},
        'target_query_labels_used_for_fitting_or_hyperparameters':False}


def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if y.shape!=p.shape or not np.isfinite(p).all():
        raise ValueError('Invalid prediction')
    e=p-y;a=abs(e)
    r={'mae':float(a.mean()),'rmse':float(np.sqrt(np.mean(e**2))),
       'p95_ae':float(np.quantile(a,.95)),'max_ae':float(a.max()),'bias':float(e.mean()),'n':len(y)}
    mask=y<.90
    r['low_soh_n']=int(mask.sum())
    r['low_soh_mae']=float(a[mask].mean()) if mask.any() else None
    r['low_soh_bias']=float(e[mask].mean()) if mask.any() else None
    return r
