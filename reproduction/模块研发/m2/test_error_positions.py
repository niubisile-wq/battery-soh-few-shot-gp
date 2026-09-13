import numpy as np
from diagnose_error_positions import stages,summarize


def test_diagnostic_stages_cover_query_positions():
    np.testing.assert_array_equal(stages(np.arange(6),6),[0,0,1,1,2,2])


def test_diagnostic_aggregation_keeps_equal_domains_and_outer_folds():
    rows=[dict(dataset='d',fold='f1',domain='a',error=1.),dict(dataset='d',fold='f1',domain='a',error=1.),
          dict(dataset='d',fold='f1',domain='b',error=3.),dict(dataset='d',fold='f2',domain='c',error=8.)]
    assert summarize(rows,['dataset'],['error'])[0]['error']==4.
    assert summarize(rows,['dataset'],['error'],nested=True)[0]['error']==5.
