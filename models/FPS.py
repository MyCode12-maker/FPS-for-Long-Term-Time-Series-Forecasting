import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Callable, Optional
from layers.Embed import DataEmbedding_wo_pos
from numpy.polynomial.legendre import legvander


class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, degree=0, stride=1, padding=0, dilation=1, groups=1, act=False,
                 bn=False, bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=None, dilation=1, groups=1, act=False, bn=False,
                 bias=False, dropout=0.):
        super(BasicConv2d, self).__init__()
        if padding is None:
            # 自动居中 padding（保持输出尺寸）
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):  # x: [B, C, H, W]
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def chebyshev_vander(x, degree):
    T = np.ones((len(x), degree + 1))
    if degree >= 1:
        T[:, 1] = x
    for n in range(2, degree + 1):
        T[:, n] = 2 * x * T[:, n - 1] - T[:, n - 2]
    return T


class SimplePolynomialFrequencyProjector(nn.Module):
    def __init__(self, max_order=12, basis_type='chebyshev'):
        super().__init__()
        self.max_order = max_order
        self.basis_type = basis_type.lower()
        self.basis_cache = {}

    def forward(self, data):

        B, N, L = data.shape
        device = data.device

        # Step 1: FFT
        fft_data = torch.fft.fft(data, dim=-1)
        fft_real = fft_data.real  # [B, N, L]
        fft_imag = fft_data.imag

        # Step 2: Polynomial basis generation (cached)
        if L not in self.basis_cache:
            x = np.linspace(-1, 1, L)
            if self.basis_type == 'chebyshev':
                basis = chebyshev_vander(x, self.max_order)  # [L, M]
            elif self.basis_type == 'legendre':
                basis = legvander(x, self.max_order)  # [L, M]
            else:
                raise ValueError(f"Unsupported basis type: {self.basis_type}")
            self.basis_cache[L] = torch.tensor(basis, dtype=torch.float32, device=device)

        basis_torch = self.basis_cache[L]  # [L, M]
        basis_pinv = torch.linalg.pinv(basis_torch)  # [M, L]

        # Step 3: Project coefficients
        coeff_real = torch.matmul(fft_real, basis_pinv.T)  # [B, N, M]
        coeff_imag = torch.matmul(fft_imag, basis_pinv.T)

        # Step 4: Reconstruct
        recon_real = torch.matmul(coeff_real, basis_torch.T)  # [B, N, L]
        recon_imag = torch.matmul(coeff_imag, basis_torch.T)

        return recon_real, recon_imag


def reconstructq(coeff_real, coeff_imag):
    recon_complex = coeff_real + 1j * coeff_imag  # [B, N, L]

    # Step 6: Apply Inverse FFT to get time domain signal
    reconstructed_time = torch.fft.ifft(recon_complex, dim=-1).real  # Only real part

    return reconstructed_time


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.order = configs.order

        self.d_model = 16
        self.d_models = 8

        self.dfactor = 1
        self.tfactor = 1

        self.patch_stride = 8
        self.patch_len = 16
        self.patch_num = int((self.seq_len - self.patch_len) / self.patch_stride + 2)
        self.patch_norm = nn.BatchNorm2d(self.channels)
        self.embedding_dropout1 = 0.2
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)

        self.enc_embedding = DataEmbedding_wo_pos(1, self.d_models, configs.embed, configs.freq,
                                                  configs.dropout)

        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_models, self.d_models * self.dfactor),
                                            nn.GELU(),
                                            nn.Dropout(self.embedding_dropout1),
                                            nn.Linear(self.d_models * self.dfactor, 1))

        self.embeddingMixer2 = nn.Sequential(nn.Linear(self.seq_len, self.seq_len * self.tfactor),
                                             nn.GELU(),
                                             nn.Dropout(self.embedding_dropout1),
                                             nn.Linear(self.seq_len * self.tfactor, self.pred_len))

        self.conv1 = BasicConv(self.patch_num, self.patch_num, kernel_size=3, groups=self.patch_num)
        self.conv2d = BasicConv2d(self.channels, self.channels, kernel_size=3, groups=self.channels)

        self.head = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_len))

        self.alpha = nn.Parameter(torch.tensor(0.8))

        self.PolynomialFrequencyProjector = SimplePolynomialFrequencyProjector(max_order=self.order,

                                                                               basis_type='chebyshev')

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch

    def TDT(self, x):
        B, T, N = x.shape

        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        x = self.enc_embedding(x, None)

        x = x.permute(0, 2, 1)
        x = self.embeddingMixer2(x)
        x = x.permute(0, 2, 1)
        x = self.embeddingMixer(x)
        x = x.permute(0, 2, 1)

        x = x.reshape(B, self.channels, self.pred_len).contiguous()

        return x
    def FDT(self,x):
        B, T, N = x.shape
        x = x.permute(0, 2, 1)

        x_real, x_imag = self.PolynomialFrequencyProjector(x)  # chebyshev legendre

        x_real = self.do_patching(x_real)
        x_imag = self.do_patching(x_imag)

        x_real = self.patch_norm(x_real)
        x_imag = self.patch_norm(x_imag)

        x_real = self.conv1(x_real.reshape(B * N, self.patch_num, self.patch_len).permute(0, 2, 1))
        x_imag = self.conv1(x_imag.reshape(B * N, self.patch_num, self.patch_len).permute(0, 2, 1))

        x_real = x_real.reshape(B, N, self.patch_num, self.patch_len)
        x_imag = x_imag.reshape(B, N, self.patch_num, self.patch_len)

        x_real = self.patch_embedding_layer(x_real)
        x_imag = self.patch_embedding_layer(x_imag)

        res_real = x_real
        res_imag = x_imag

        x_real = self.conv2d(x_real)
        x_imag = self.conv2d(x_imag)

        x_real = x_real + res_real
        x_imag = x_imag + res_imag

        x = reconstructq(x_real, x_imag)
        x = self.head(x)
        return x
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):

        B, T, N = x.shape
        means = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x = (x - means) / stdev

        z = self.TDT(x)
        x = self.FDT(x)

        y = (1 - self.alpha) * x + self.alpha * z

        y = y.permute(0, 2, 1)
        y = y * stdev + means

        return y  # to [Batch, Output length, Channel]
