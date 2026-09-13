"""Frozen parent plus conservatively scaled reference-paired waveform GP branch."""
import strict_ablation as sa


class WaveformM3Adapter:
    def __init__(self,parent,branch,gain,parent_mode=None):
        self.parent,self.branch,self.gain,self.parent_mode=parent,branch,float(gain),parent_mode

    def predict(self,cell):
        c=sa.inference_view(cell)
        p=self.parent.predict(c) if self.parent_mode is None else self.parent.predict(c,self.parent_mode)
        q=self.branch.predict(c)
        return p+self.gain*(q-p)
