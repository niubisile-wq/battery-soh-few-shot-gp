from dataclasses import replace
import numpy as np
import nested_expert_probe as ne
from test_m1 import toy


def test_support_residual_never_reads_query_labels(monkeypatch):
    c=toy(0,cid='a');records=[dict(cell_id='a',base=np.array([.9,.8]))]
    def prior(model,support):
        assert len(support.y)==ne.ns.rd.sa.K
        return np.zeros(len(support.y))
    monkeypatch.setattr(ne.ns.rd.sa,'prior',prior)
    models={'Base':object(),'Base+M1':object()}
    a=ne.enrich(records,records,models,{'a':c},'Base+M1')
    y=c.y.copy();y[ne.ns.rd.sa.K:]=np.nan
    b=ne.enrich(records,records,models,{'a':replace(c,y=y)},'Base+M1')
    assert a[0]['residual']==b[0]['residual']
    assert a[0]['raw_residual']==b[0]['raw_residual']
