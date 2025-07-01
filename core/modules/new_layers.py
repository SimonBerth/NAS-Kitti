import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torchsparse.nn as spnn

from core.modules.dynamic_op import DynamicBatchNorm, DynamicLinear
from core.modules.dynamic_sparseop import (SparseDynamicBatchNorm,
                                           SparseDynamicConv3d, make_divisible)
from core.modules.modules import RandomDepth, RandomModule


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
    bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


class LinearBlock(nn.Module):

    def __init__(self, inc, outc, bias=True, no_relu=False, no_bn=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.no_relu = no_relu
        self.bias = bias
        self.no_bn = no_bn
        net = OrderedDict([('conv', nn.Linear(inc, outc, bias=bias))])
        if not self.no_bn:
            net['bn'] = nn.BatchNorm1d(outc)
        if not self.no_relu:
            net['act'] = nn.ReLU(True)

        self.net = nn.Sequential(net)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, nas_module, runtime_inc_constraint=None):
        cur_kernel = nas_module.net.conv.linear.weight
        if runtime_inc_constraint is None:
            cur_kernel = cur_kernel[:, :self.inc]
        else:
            cur_kernel = cur_kernel[:, runtime_inc_constraint]
        cur_kernel = cur_kernel[:self.outc, :]
        self.net.conv.weight.data = cur_kernel

        if self.bias:
            cur_bias = nas_module.net.conv.linear.bias
            cur_bias = cur_bias[:self.outc]
            self.net.conv.bias.data = cur_bias

        if not self.no_bn:
            self.net.bn.weight.data = nas_module.net.bn.bn.weight[:self.outc]
            self.net.bn.running_var.data = nas_module.net.bn.bn.running_var[:
                                                                            self
                                                                            .
                                                                            outc]
            self.net.bn.running_mean.data = nas_module.net.bn.bn.running_mean[:
                                                                              self
                                                                              .
                                                                              outc]
            self.net.bn.bias.data = nas_module.net.bn.bn.bias[:self.outc]
            self.net.bn.num_batches_tracked.data = \
                nas_module.net.bn.bn.num_batches_tracked

    def forward(self, inputs):
        return self.net(inputs)


class DynamicLinearBlock(RandomModule):

    def __init__(self,
                 inc,
                 outc,
                 cr_bounds=(0.25, 1.0),
                 bias=True,
                 no_relu=False,
                 no_bn=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.bias = bias
        self.cr_bounds = cr_bounds
        self.no_relu = no_relu
        self.no_bn = no_bn

        net = OrderedDict([('conv', DynamicLinear(inc, outc, bias=bias))])
        if not self.no_bn:
            net['bn'] = DynamicBatchNorm(outc)
        if not self.no_relu:
            net['act'] = nn.ReLU(True)

        self.net = nn.Sequential(net)
        self.runtime_inc = None
        self.runtime_outc = None
        self.in_channel_constraint = None

    def re_organize_middle_weights(self):
        weights = self.net.conv.linear.weight.data
        outc, inc = weights.shape
        importance = torch.sum(torch.abs(weights), dim=(1))

        sorted_importance, sorted_idx = torch.sort(importance,
                                                   dim=0,
                                                   descending=True)
        self.net.conv.linear.weight.data = torch.index_select(
            self.net.conv.linear.weight.data, 0, sorted_idx)
        if self.bias:
            self.net.conv.linear.bias.data = torch.index_select(
                self.net.conv.linear.bias.data, 0, sorted_idx)
        adjust_bn_according_to_idx(self.net.bn.bn, sorted_idx)

    def constrain_in_channel(self, constraint):
        self.in_channel_constraint = constraint
        self.runtime_inc = None

    def manual_select(self, channel):
        self.net.conv.set_output_channel(channel)
        if not self.no_bn:
            self.net.bn.set_channel(channel)
        self.runtime_outc = channel

    def manual_select_in(self, channel):
        self.runtime_inc = channel

    def random_sample(self):
        cr = random.uniform(*self.cr_bounds)
        channel = make_divisible(int(cr * self.outc))
        self.net.conv.set_output_channel(channel)
        if not self.no_bn:
            self.net.bn.set_channel(channel)
        self.runtime_outc = channel
        return channel

    def clear_sample(self):
        self.runtime_outc = None

    def status(self):
        return self.runtime_outc

    def determinize(self):
        assert self.runtime_inc is not None or self.in_channel_constraint is not None

        inc = self.runtime_inc if self.runtime_inc is not None \
            else len(self.in_channel_constraint)

        determinized_model = LinearBlock(inc,
                                         self.runtime_outc,
                                         bias=self.bias,
                                         no_relu=self.no_relu,
                                         no_bn=self.no_bn)
        determinized_model.load_weights(self, self.in_channel_constraint)
        return determinized_model

    def forward(self, x):
        if self.in_channel_constraint is None:
            in_channel = x.shape[-1]
            self.runtime_inc = in_channel
            self.net.conv.set_in_channel(in_channel=in_channel)
        else:
            self.net.conv.set_in_channel(constraint=self.in_channel_constraint)
        out = self.net(x)
        return out


class ConvolutionBlock(nn.Module):

    def __init__(self,
                 inc,
                 outc,
                 ks=3,
                 stride=1,
                 dilation=1,
                 no_relu=False,
                 transposed=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        self.no_relu = no_relu
        self.net = nn.Sequential(
            OrderedDict([
                ('conv',
                 spnn.Conv3d(inc,
                             outc,
                             kernel_size=ks,
                             dilation=dilation,
                             stride=stride,
                             transposed=transposed)),
                ('bn', spnn.BatchNorm(outc)),
                ('act',
                 spnn.ReLU(True) if not self.no_relu else nn.Sequential())
            ]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, nas_module, runtime_inc_constraint=None):
        cur_kernel = nas_module.net.conv.kernel
        if runtime_inc_constraint is not None:
            if self.ks > 1:
                cur_kernel = cur_kernel[:, runtime_inc_constraint, :]
            else:
                cur_kernel = cur_kernel[runtime_inc_constraint]
        else:
            if self.ks > 1:
                cur_kernel = cur_kernel[:, torch.arange(self.inc), :]
            else:
                cur_kernel = cur_kernel[torch.arange(self.inc)]

        cur_kernel = cur_kernel[..., torch.arange(self.outc)]
        self.net.conv.kernel.data = cur_kernel
        self.net.bn.weight.data = nas_module.net.bn.bn.weight[:self.outc]
        self.net.bn.running_var.data = nas_module.net.bn.bn.running_var[:self.
                                                                        outc]
        self.net.bn.running_mean.data = nas_module.net.bn.bn.running_mean[:self.
                                                                          outc]
        self.net.bn.bias.data = nas_module.net.bn.bn.bias[:self.outc]
        self.net.bn.num_batches_tracked.data = \
            nas_module.net.bn.bn.num_batches_tracked

    def forward(self, inputs):
        return self.net(inputs)


class DynamicConvolutionBlock(RandomModule):

    def __init__(self,
                 inc,
                 outc,
                 cr_bounds=(0.25, 1.0),
                 ks=3,
                 stride=1,
                 dilation=1,
                 no_relu=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        self.s = stride
        self.cr_bounds = cr_bounds
        self.no_relu = no_relu
        self.net = nn.Sequential(
            OrderedDict([
                ('conv',
                 SparseDynamicConv3d(inc,
                                     outc,
                                     kernel_size=ks,
                                     dilation=dilation,
                                     stride=stride)),
                ('bn', SparseDynamicBatchNorm(outc)),
                ('act',
                 spnn.ReLU(True) if not self.no_relu else nn.Sequential())
            ]))
        self.runtime_inc = None
        self.runtime_outc = None
        self.in_channel_constraint = None

    def re_organize_middle_weights(self):
        weights = self.net.conv.kernel.data
        if len(weights.shape) == 3:
            k, inc, outc = weights.shape
            importance = torch.sum(torch.abs(weights), dim=(0, 1))
        else:
            inc, outc = weights.shape
            importance = torch.sum(torch.abs(weights), dim=(0))

        sorted_importance, sorted_idx = torch.sort(importance,
                                                   dim=0,
                                                   descending=True)
        if len(weights.shape) == 3:
            self.net.conv.kernel.data = torch.index_select(
                self.net.conv.kernel.data, 2, sorted_idx)
        else:
            self.net.conv.kernel.data = torch.index_select(
                self.net.conv.kernel.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.net.bn.bn, sorted_idx)

    def constrain_in_channel(self, constraint):
        self.in_channel_constraint = constraint
        self.runtime_inc = None

    def manual_select(self, channel):
        self.net.conv.set_output_channel(channel)
        self.net.bn.set_channel(channel)
        self.runtime_outc = channel

    def manual_select_in(self, channel):
        if self.in_channel_constraint is not None:
            return
        self.runtime_inc = channel

    def random_sample(self):
        cr = random.uniform(*self.cr_bounds)
        channel = make_divisible(int(cr * self.outc))
        self.net.conv.set_output_channel(channel)
        self.net.bn.set_channel(channel)
        self.runtime_outc = channel
        return channel

    def clear_sample(self):
        self.runtime_outc = None

    def status(self):
        return self.runtime_outc

    def determinize(self):
        if self.runtime_inc is None:
            assert self.in_channel_constraint is not None
            inc = len(self.in_channel_constraint)
        else:
            inc = self.runtime_inc

        determinized_model = ConvolutionBlock(inc,
                                              self.runtime_outc,
                                              self.ks,
                                              self.s,
                                              no_relu=self.no_relu)
        determinized_model.load_weights(self, self.in_channel_constraint)
        return determinized_model

    def forward(self, x):
        if self.in_channel_constraint is None:
            in_channel = x.F.shape[-1]
            self.runtime_inc = in_channel
            self.net.conv.set_in_channel(in_channel=in_channel)
        else:
            self.net.conv.set_in_channel(constraint=self.in_channel_constraint)

        out = self.net(x)
        return out

class SPVConvBlock(RandomModule):
    def __init__(self, in_channel, out_channel, stride=1, padding=0, norm_fn=None,)
                 

class DynamicSPVConvBlock(RandomModule):
    # Search space : number of features, random depth for Linear and Conv3d
    def __init__(self, in_channel, out_channel, stride=1, padding=0, norm_fn=None,
                 possible_ks=((3,3,3),(5,5,5)),
                 cr_bounds=(0.25, 1.0), 
                 depth_mlp=(0, 5), 
                 depth_conv=(1, 3)):
        super().__init__()

        self.inc = in_channel
        self.outc = out_channel
        self.s = stride
        self.pad = padding

        self.d_min_mlp  = depth_mlp[0]
        self.d_max_mlp  = depth_mlp[1]
        self.d_min_conv = depth_conv[0]
        self.d_max_conv = depth_conv[1]
        self.cr_bounds = cr_bounds
        self.possible_ks = possible_ks


        [
            DynamicResidualBlock(base_channels,
                                 output_channels[i],
                                 cr_bounds=self.cr_bounds,
                                 ks=3,
                                 stride=1,
                                 dilation=1),
            DynamicResidualBlock(output_channels[i],
                                 output_channels[i],
                                 cr_bounds=self.cr_bounds,
                                 ks=3,
                                 stride=1,
                                 dilation=1)
        ]

        '''
        self.layers = nn.ModuleList(layers)

        self.conv = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            norm_fn(out_channels),
            spnn.ReLU()
        )
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, out_channels), 
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        '''


DynamicLinearBlock(output_channels[0],
                    output_channels[num_down_stages],
                    bias=True,
                    no_relu=False,
                    no_bn=False),
class DynamicLinearBlock(RandomModule):
    def __init__(self,
                 inc,
                 outc,
                 cr_bounds=(0.25, 1.0),
                 bias=True,
                 no_relu=False,
                 no_bn=False):

DynamicConvolutionBlock(base_channels,
                        base_channels,
                        cr_bounds=self.trans_cr_bounds,
                        ks=2,
                        stride=2,
                        dilation=1)
class DynamicConvolutionBlock(RandomModule):
    def __init__(self,
                 inc,
                 outc,
                 cr_bounds=(0.25, 1.0),
                 ks=3,
                 stride=1,
                 dilation=1,
                 no_relu=False):

'''
    def random_sample(self):
        if self.depth_min is not None:
            depth_min = self.depth_min
        else:
            depth_min = 0

        if self.depth_max is not None:
            depth_max = self.depth_max
        else:
            depth_max = len(self.layers)

        self.depth = random.randint(depth_min, depth_max)
        return self.depth

    def clear_sample(self):
        self.depth = None

    def status(self):
        return self.depth

    def manual_select(self, depth):
        self.depth = depth
'''

    def forward(self, voxel, points):
        p = voxel_to_point(voxel,points)
        p.F = self.MLP(p.F)
        out = self.conv(voxel)
        out.F = out.F + point_to_voxel(out, p).F
        return out

'''
    # fixme: support tuples as input
    def forward(self, x):
        for k in range(self.depth):
            x = self.layers[k](x)
        return x


    def determinize(self):
        return nn.Sequential(*self.layers[:self.depth])
'''


