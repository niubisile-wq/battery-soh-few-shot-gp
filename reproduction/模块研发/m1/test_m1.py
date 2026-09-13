import copy
import sys
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from sklearn.base import clone
from scipy.linalg import cho_solve
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import Cell,K,source_indices,cap_training_cells,split,load_cells
from features import raw_view,FeatureMap
from gp import make_kernel,GPModel
from screen import SPECS,macro


def toy(seed=0,n=24,cid='a',domain='a'):
    rng=np.random.default_rng(seed);t=np.linspace(0,1,128)
    health=np.linspace(1,.85,n)
    duration=100+health*100
    v=3.9+.2*t[None,:]+.002*rng.normal(size=(n,128));v[:,0]=3.9;v[:,-1]=4.1
    x=np.stack([v,np.full((n,128),2.),duration[:,None]*t],1).astype('float32')
    return Cell(cid,'toy',domain,x,health.astype('float32'),np.arange(n,dtype='float32'),2.,2.)


def test_causal_reference_views_and_scaler():
    train=[toy(cid='a'),toy(1,cid='b')];idx=source_indices(train)
    for name in ['C00_original','C02_pruned','C05_physics','P02_dual','P02_anchor_physics','P04_pls_metric']:
        spec=SPECS[name];fm=FeatureMap(spec,spec['configs'][0]).fit(train,idx)
        target=toy(3,cid='target');changed=copy.deepcopy(target)
        changed.x[18:]*=1.2;changed.y[K:]=42;changed.cycle[K:]*=100
        assert_allclose(fm.transform(target)[:18],fm.transform(changed)[:18],atol=0,rtol=0)
        changed.y[:K]=.1
        assert_allclose(fm.transform(target)[:18],fm.transform(changed)[:18],atol=0,rtol=0)


def test_group_kernel_psd_clone_gradient():
    rng=np.random.default_rng(4);x=rng.normal(size=(7,21))
    kernel=make_kernel('group',1.,21);other=clone(kernel)
    k,g=other(x,eval_gradient=True)
    assert_allclose(k,k.T,atol=1e-12)
    assert np.linalg.eigvalsh(k).min()>-1e-10
    for j in range(len(kernel.theta)):
        th=kernel.theta.copy();th[j]+=1e-5
        kp=kernel.clone_with_theta(th)(x)
        th[j]-=2e-5;km=kernel.clone_with_theta(th)(x)
        assert_allclose(g[:,:,j],(kp-km)/2e-5,atol=1e-7,rtol=1e-5)


def test_source_cap_and_budget():
    cells=[toy(cid=f'c{i}',domain=str(i%4)) for i in range(130)]
    tr,va,te=split(cells,'toy:0')
    assert len(tr)==60
    idx=source_indices(tr)
    assert sum(map(len,idx.values()))<=1000
    assert all(set(range(K))<=set(a) for a in idx.values())
    assert not ({c.id for c in tr}&{c.id for c in te})
    assert [c.id for c in cap_training_cells(cells)]==[c.id for c in cap_training_cells(list(reversed(cells)))]


def test_posterior_update_matches_gaussian_conditioning():
    cells=[toy(0,cid='a'),toy(1,cid='b')];idx=source_indices(cells)
    spec=SPECS['C01_support_update'];model=GPModel(spec,spec['configs'][0]).fit(cells,idx)
    target=toy(2,cid='t');z=model.features.transform(target)
    mean,cov=model.gp.predict(z,return_cov=True)
    # Implementation uses a tiny numerical jitter in normalized GP units.
    jitter=1e-9*float(model.gp._y_train_std)**2
    expected=mean[K:]+cov[K:,:K]@np.linalg.solve(cov[:K,:K]+np.eye(K)*jitter,target.y[:K]-mean[:K])
    assert_allclose(model.predict(target,'posterior'),expected,atol=1e-8)
    changed=copy.deepcopy(target);changed.y[K:]=-100
    assert_allclose(model.predict(changed,'posterior'),expected,atol=1e-8)


def test_xjtu_original_source_keys():
    path=Path(__file__).resolve().parents[2]/'开发基线选择依据/results/fair_selection_v1/jobs/GPR__3C__s0/provenance.json'
    import json
    old=json.loads(path.read_text())
    tr,va,te=split(load_cells('XJTU'),'XJTU:3C');idx=source_indices(tr)
    keys=[[c.id,float(c.cycle[j])] for c in tr for j in idx[c.id]]
    assert keys==old['source_sample_keys']
    assert {c.id for c in va}==set(old['tune_cells'])
    assert {c.id for c in te}==set(old['development_target_cells'])


def test_equal_domain_macro():
    rows=[{'dataset':'a','domain':'x','mae':.01}]*20+[{'dataset':'a','domain':'y','mae':.09}]
    assert_allclose(macro(rows),.05)
