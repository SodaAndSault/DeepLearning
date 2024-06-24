import torch
import scipy.io
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import relu
import torch
from torch.nn.functional import max_pool3d, avg_pool3d, dropout, dropout3d, interpolate
from torch import tanh, relu, sigmoid
from typing import Optional
from torch.nn.functional import (
    avg_pool2d,
    dropout,
    dropout2d,
    interpolate,
    max_pool2d,
    relu,
    sigmoid,
    tanh,
)


def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

#复数2D卷积
class ComplexConv2d(torch.nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self,input):    
        return apply_complex(self.conv_r, self.conv_i, input)
    
#复数线性层
class ComplexLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = torch.nn.Linear(in_features, out_features)
        self.fc_i = torch.nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)
    
#复数反卷积层
class ComplexConvTranspose2d(torch.nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
    
    def forward(self, input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)
    
def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)



# 依托pytorch自带的BatchNorm实现
class NaiveComplexBatchNorm2d(nn.Module):
    """
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, inp):
        return self.bn_r(inp.real).type(torch.complex64) + 1j * self.bn_i(
            inp.imag
        ).type(torch.complex64)


# 自行实现3D ComplexBatchNorm
class _ComplexBatchNorm(nn.Module):
    running_mean: Optional[torch.Tensor]

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.complex64)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)
            

class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, inp):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = inp.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = inp.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, :, None, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * inp.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)  #
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (
            Rrr[None, :, None, None] * inp.real + Rri[None, :, None, None] * inp.imag
        ).type(torch.complex64) + 1j * (
            Rii[None, :, None, None] * inp.imag + Rri[None, :, None, None] * inp.real
        ).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                self.weight[None, :, 0, None, None] * inp.real
                + self.weight[None, :, 2, None, None] * inp.imag
                + self.bias[None, :, 0, None, None]
            ).type(torch.complex64) + 1j * (
                self.weight[None, :, 2, None, None] * inp.real
                + self.weight[None, :, 1, None, None] * inp.imag
                + self.bias[None, :, 1, None, None]
            ).type(
                torch.complex64
            )
        return inp

# 复数2D MaxPooling
def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-3)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-3)
    ).view_as(indices)
    return output
    
def complex_max_pool2d(
    inp,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """
    Perform complex max pooling by selecting on the absolute value on the complex values.
    """
    # Calculate absolute value and indices using 3D max pooling
    absolute_value, indices = max_pool2d(
        inp.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )

    # Convert absolute value to complex64
    absolute_value = absolute_value.type(torch.complex64)

    # Retrieve the phase information using indices
    angle = torch.atan2(inp.imag, inp.real)
    angle = _retrieve_elements_from_indices(angle, indices)

    # Reconstruct the complex values using absolute value and phase information
    result = absolute_value * (
        torch.cos(angle).type(torch.complex64)
        + 1j * torch.sin(angle).type(torch.complex64)
    )

    if return_indices:
        return result, indices
    else:
        return result

class ComplexMaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool2d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
    
# 复数 上采样
def complex_upsample(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the amplitude and phase part and recombining
    """
    outp_abs = interpolate(
        inp.abs(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    angle = torch.atan2(inp.imag, inp.real)
    outp_angle = interpolate(
        angle,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return outp_abs * (
        torch.cos(outp_angle).type(torch.complex64)
        + 1j * torch.sin(outp_angle).type(torch.complex64)
    )