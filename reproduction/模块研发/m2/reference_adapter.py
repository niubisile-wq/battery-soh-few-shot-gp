"""Frozen M2 reference-risk inference adapter: only K support labels visible."""
import strict_ablation as sa
import reference_gate as rg


class ReferenceRiskAdapter:
    def __init__(self,parent,physical,parent_mode,physical_mode,head,selection,raw_model=None,raw_mode=None):
        self.parent=parent;self.physical=physical
        self.parent_mode=parent_mode;self.physical_mode=physical_mode
        self.head=head;self.selection=selection
        self.raw_model=raw_model;self.raw_mode=raw_mode
        self.head.setdefault('params',rg.PARAMS)
        self.head.setdefault('settings',rg.SETTINGS)

    def predict(self,cell):
        c=sa.inference_view(cell)
        base,residual=sa.predict_components(self.parent,c,self.parent_mode)
        physical=self.physical.predict(c,self.physical_mode)
        episode=dict(base=base,residual=residual,physical=physical)
        if self.head.get('raw_reference',False):
            raw,r=sa.predict_components(self.raw_model,c,self.raw_mode)
            episode.update(raw_base=raw,raw_residual=r)
        e=rg.attach(episode,c,self.head)
        return e['reference_predictions'][self.selection['reference_view']][self.selection['reference_index']]
