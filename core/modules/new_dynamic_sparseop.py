import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse.nn.functional as spf
from torchsparse import SparseTensor

__all__ = ["get_slice", "SparseDynamicConv3dKer", "SparseDynamicBatchNorm"]

# Rounds to nearest mutliple of 16, instead of flooring to it.
# Makes last 16 channels available in practice.
def make_divisible(x):
    return max(16, int(round(x / 16) * 16))

def get_slice(total_size, sub_size):
    start = (total_size - sub_size) // 2
    return slice(start, start + sub_size)

# Modified version of SparseDynamicConv3d for variable kernel size.
# This is useful for dynamic convolution where the kernel size can change.
class SparseDynamicConv3dKer(nn.Module):
    # padding=None for automatic padding calculation based on kernel size.
    def __init__(self, inc, outc, kernel_size=(5,5,5), stride=1, padding=None, transposed=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = kernel_size
        if isinstance(kernel_size, int):
            self.ks = (kernel_size, kernel_size, kernel_size)
        self.k = self.ks[0] * self.ks[1] * self.ks[2]
        self.s = stride
        self.padding = padding
        self.kernel = (
            nn.Parameter(torch.zeros(self.k, inc, outc)) if self.k > 1 else nn.Parameter(torch.zeros(inc, outc))
        )
        self.t = transposed
        self.init_weight()
        self.runtime_outc = None
        self.runtime_inc = None
        self.runtime_inc_constraint = None
        self.runtime_ks = None
        if kernel_size == 1:
            assert not transposed

    def __repr__(self):
        if not self.t:
            return "SparseDynamicConv3dKer(imax=%s, omax=%s, ks=%s, s=%s)" % (self.inc, self.outc, self.ks, self.s)
        else:
            return "SparseDynamicConv3dKerTranspose(imax=%s, omax=%s, ks=%s, s=%s)" % (self.inc, self.outc, self.ks, self.s)

    def init_weight(self):
        std = 1.0 / math.sqrt(self.outc if self.t else self.inc * self.k)
        self.kernel.data.uniform_(-std, std)

    def set_in_channel(self, in_channel=None, constraint=None):
        if in_channel is not None:
            self.runtime_inc = in_channel
        elif constraint is not None:
            self.runtime_inc_constraint = torch.from_numpy(np.array(constraint)).long()

    def set_output_channel(self, out_channel):
        self.runtime_outc = out_channel

    def set_kernel_size(self, kernel_size):
        if isinstance(kernel_size, int):
            self.runtime_ks = (kernel_size, kernel_size, kernel_size)
        else:
            self.runtime_ks = kernel_size

    def set_padding(self, padding):
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding

    def forward(self, inputs):
        # inputs: SparseTensor
        # outputs: SparseTensor
        features = inputs.F
        inputs.C
        inputs.s
        cur_kernel = self.kernel

        if self.runtime_ks is not None:
            cur_kernel = cur_kernel.view(self.ks[2], self.ks[1], self.ks[0], self.inc, self.outc)
            cur_kernel = cur_kernel[get_slice(self.ks[2], self.runtime_ks[2]), 
                                    get_slice(self.ks[1], self.runtime_ks[1]), 
                                    get_slice(self.ks[0], self.runtime_ks[0]), :, :]
            cur_kernel = cur_kernel.contiguous().view(-1, self.inc, self.outc)
        
        else:
            assert 0, print("Kernel size not specified!")

        if self.padding is None:
            if self.t:
                padding = (self.runtime_ks[0] - 1) // 2, (self.runtime_ks[1] - 1) // 2, (self.runtime_ks[2] - 1) // 2
            else:
                padding = (self.runtime_ks[0] // 2, self.runtime_ks[1] // 2, self.runtime_ks[2] // 2)
            self.padding = padding

        if self.runtime_inc_constraint is not None:
            self.runtime_inc_constraint = self.runtime_inc_constraint.to(features.device)

            cur_kernel = (
                cur_kernel[:, self.runtime_inc_constraint, :]
                if self.ks > (1,1,1)
                else cur_kernel[self.runtime_inc_constraint]
            )

        elif self.runtime_inc is not None:
            cur_kernel = cur_kernel[:, : self.runtime_inc, :] if self.ks > (1,1,1) else cur_kernel[: self.runtime_inc]

        else:
            assert 0, print("Number of channels not specified!")

        cur_kernel = cur_kernel[..., : self.runtime_outc]

        # 4th argument -> bias
        return spf.conv3d(inputs, cur_kernel, self.runtime_ks, None, stride=self.s, padding=self.padding, transposed=self.t)


class SparseDynamicBatchNorm(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, c, cr_bounds=[0.25, 1.0], eps=1e-3, momentum=0.01):
        super().__init__()
        self.c = c
        self.eps = eps
        self.momentum = momentum
        self.cr_bounds = cr_bounds
        self.bn = nn.BatchNorm1d(c, eps=eps, momentum=momentum)
        self.channels = []
        self.runtime_channel = None

    def __repr__(self):
        return "SparseDynamicBatchNorm(cmax=%d)" % self.c

    def set_channel(self, channel):
        self.runtime_channel = channel

    def bn_foward(self, x, bn, feature_dim):
        if bn.num_features == feature_dim or SparseDynamicBatchNorm.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, inputs):
        output_features = self.bn_foward(inputs.F, self.bn, inputs.F.shape[-1])
        output_tensor = SparseTensor(
            coords=inputs.coords,
            feats=output_features,
            stride=inputs.stride,
            spatial_range=inputs.spatial_range,
        )
        output_tensor = SparseTensor(output_features, inputs.C, inputs.s)
        output_tensor._caches = inputs._caches

        return output_tensor
