from dataclasses import dataclass,replace
import numpy as np
from correction import ResidualCorrector,features


@dataclass
class Cell:
    id:str
    domain:str
    x:np.ndarray
    y:np.ndarray
    cycle:np.ndarray


def fixture():
    rng=np.random.default_rng(9);cells={};episodes=[]
    for i in range(4):
        x=rng.normal(size=(20,3,128));x[:,0]+=3.7;x[:,1]+=2;x[:,2]=np.arange(128)
        y=np.linspace(1,.8,20);c=Cell(str(i),str(i%2),x,y,np.arange(20));cells[c.id]=c
        q=np.arange(10,20);p=y[q]+(.01 if i%2 else .03)
        episodes.append(dict(cell_id=c.id,domain=c.domain,query_indices=q,y=y[q],predictions={'B123':p}))
    return cells,episodes


def test_constant_and_budget():
    cells,ee=fixture();m=ResidualCorrector().fit(ee,cells,'B123')
    np.testing.assert_allclose(m.mean,-.02)
    assert len(m.source_keys)==40 and all(j>=10 for cid,j in m.source_keys)


def test_label_prefix_and_clip():
    cells,ee=fixture();c=cells['0'];p=ee[0]['predictions']['B123']
    for view in ['state','signal']:
        for learner in ['ridge','rbf']:
            m=ResidualCorrector(view,learner,10).fit(ee,cells,'B123');r=m.predict(c,p)
            yy=c.y.copy();yy[10:]=456
            np.testing.assert_allclose(r,m.predict(replace(c,y=yy),p),atol=1e-10)
            short=replace(c,x=c.x[:13],y=c.y[:13],cycle=c.cycle[:13])
            np.testing.assert_allclose(r[:3],m.predict(short,p[:3]),atol=1e-10)
            assert np.isfinite(r).all() and abs(r).max()<=.05


def test_training_ignores_nonbudget_cell_labels():
    cells,ee=fixture();hidden={}
    for cid,c in cells.items():
        y=c.y.copy();y[10:]=np.nan;hidden[cid]=replace(c,y=y)
    for learner in ['constant','ridge','rbf']:
        a=ResidualCorrector('signal',learner,10).fit(ee,cells,'B123')
        b=ResidualCorrector('signal',learner,10).fit(ee,hidden,'B123')
        for e in ee:
            c=cells[e['cell_id']];pred=e['predictions']['B123']
            np.testing.assert_allclose(a.predict(c,pred),b.predict(c,pred),atol=1e-10)


def test_full_ablation_edges():
    from score import addition_edges
    edges=addition_edges()
    assert len(edges)==len(set(edges))==32
    assert sum('4' not in b and '4' in a for a,b in edges)==8
    assert {b for a,b in edges if a=='B1234'}=={'B123','B124','B134','B234'}
    for a,b in edges:
        assert set(a[1:])>set(b[1:]) and len(a)==len(b)+1
