import numpy as np
import residual_head as rh


def episode(domain):
    b=np.linspace(.6,1.,30)
    return dict(domain=domain,base=b,physical=b-.02,residual=.01,
                parent_support_mse=.001,physical_support_mse=.002,
                parent_variance=np.full(30,.003),physical_variance=np.full(30,.004),
                y=b+.05*b-.01)


def test_learned_head_recovers_source_residual_and_does_not_read_query_labels():
    heads=rh.fit([episode('a'),episode('b')])
    e=episode('held');y=e.pop('y')
    p=rh.attach(e,heads)['meta_predictions']['parent'][0]
    np.testing.assert_allclose(p,y-e['base'],atol=1e-3)
    e['y']=np.full(30,123.)
    np.testing.assert_array_equal(rh.attach(e,heads)['meta_predictions']['parent'][0],p)


def test_features_and_residual_head_are_query_prefix_causal():
    heads=rh.fit([episode('a'),episode('b')]);e=episode('held')
    short={k:(v[:3] if isinstance(v,np.ndarray) else v) for k,v in e.items()}
    full=rh.attach(e,heads)['meta_predictions'];prefix=rh.attach(short,heads)['meta_predictions']
    for kind in ['parent','dual']:
        np.testing.assert_allclose(prefix[kind],full[kind][:,:3],atol=1e-12)
