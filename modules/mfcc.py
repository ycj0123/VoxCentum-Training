import torch
from torchaudio import transforms as T


class MFCC_Delta(torch.nn.Module):
    def __init__(self, mfcckwargs={}, deltakwargs={}) -> None:
        super().__init__()
        self.mfcc = T.MFCC(**mfcckwargs)
        self.delta = T.ComputeDeltas(**deltakwargs)

    def forward(self, x):
        x = self.mfcc(x)
        delta = self.delta(x)
        delta2 = self.delta(delta)
        output = torch.cat((x, delta, delta2), 1)
        return output
