from dataclasses import replace
import numpy as np
import diagnose_cap_routing as dc
from test_m1 import toy


def test_identical_heads_make_all_stages_identical_when_cap_inactive():
    cells=[toy(i,cid=str(i),domain=str(i)) for i in range(3)]
    episodes=[dict(cell_id=c.id,domain=c.domain,base=np.array([.9,.8]),physical=np.array([.85,.75]),residual=0.,y=np.array([.88,.78])) for c in cells]
    head=dc.rg.fit(episodes,cells);sel=dict(reference_view='hi',reference_index=0)
    outputs=dc.stage_predictions(episodes[0],cells[0],head,dict(head,physical_cap=True),sel,sel)
    for p in outputs.values():np.testing.assert_array_equal(p,outputs['v8'])
