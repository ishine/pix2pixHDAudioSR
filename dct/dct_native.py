from turtle import forward
import torch
import torch.nn.functional as F
from .discrete_spectral_transform import get_expk
from . import torch_fft_api as torch_fft_api

class DCT_2N_native(torch.nn.Module):
    def __init__(self, expk=None):
        super(DCT_2N_native, self).__init__()

        self.expk = expk
        self.out = None
        self.buf = None
        self.N = None

    def forward(self, x):
        # last dimension
        if self.N is None:
            self.N = x.size(-1)
        # pad last dimension
        x_pad = F.pad(x, (0, self.N), 'constant', 0)
        # the last dimension here becomes -2 because complex numbers introduce a new dimension
        y = torch_fft_api.rfft(x_pad, signal_ndim=1, normalized=False, onesided=True)[..., 0:self.N, :]
        y.mul_(1.0/self.N)

        if self.expk is None:
            self.expk = get_expk(self.N, dtype=x.dtype, device=x.device)

        # get real part
        y.mul_(self.expk)

        # I found add is much faster than sum
        #y = y.sum(dim=-1)
        return y[..., 0]+y[..., 1]

class IDCT_2N_native(torch.nn.Module):
    def __init__(self, expk=None):
        super(IDCT_2N_native, self).__init__()

        self.expk = expk
        self.out = None
        self.buf = None
        self.N = None

    def forward(self,x):
        # last dimension
        if self.N is None:
            self.N = x.size(-1)

        if self.expk is None:
            self.expk = get_expk(self.N, dtype=x.dtype, device=x.device)

        # multiply by 2*exp(1j*pi*u/(2N))
        x_pad = x.unsqueeze(-1).mul(self.expk)
        # pad second last dimension, excluding the complex number dimension
        x_pad = F.pad(x_pad, (0, 0, 0, self.N), 'constant', 0)

        if len(x.size()) == 1:
            x_pad.unsqueeze_(0)

        # the last dimension here becomes -2 because complex numbers introduce a new dimension
        y = torch_fft_api.irfft(x_pad, signal_ndim=1, normalized=False, onesided=False, signal_sizes=[2*self.N])[..., 0:self.N]
        y.mul_(self.N)

        if len(x.size()) == 1:
            y.squeeze_(0)

        return y