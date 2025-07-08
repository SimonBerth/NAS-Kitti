import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torchsparse.nn as spnn

from core.modules.dynamic_op import DynamicBatchNorm, DynamicLinear
from core.modules.modules import RandomDepth, RandomModule

from core.modules.new_dynamic_sparseop import (SparseDynamicConv3dKer, SparseDynamicBatchNorm,
                                               make_divisible, get_slice)

from core.models.utils import  voxel_to_point, point_to_voxel


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

        self.mlp = nn.Sequential(net)
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
        self.mlp.conv.weight.data = cur_kernel

        if self.bias:
            cur_bias = nas_module.net.conv.linear.bias
            cur_bias = cur_bias[:self.outc]
            self.mlp.conv.bias.data = cur_bias

        if not self.no_bn:
            self.mlp.bn.weight.data = nas_module.net.bn.bn.weight[:self.outc]
            self.mlp.bn.running_var.data = nas_module.net.bn.bn.running_var[:self.outc]
            self.mlp.bn.running_mean.data = nas_module.net.bn.bn.running_mean[:self.outc]
            self.mlp.bn.bias.data = nas_module.net.bn.bn.bias[:self.outc]
            self.mlp.bn.num_batches_tracked.data = \
                nas_module.net.bn.bn.num_batches_tracked

    def forward(self, inputs):
        return self.mlp(inputs)



class ConvolutionBlock(nn.Module):

    def __init__(self,
                 inc,
                 outc,
                 ks=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 no_relu=False,
                 transposed=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        if isinstance(ks, int):
            self.ks = (ks, ks, ks)
        self.no_relu = no_relu
        self.conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 spnn.Conv3d(inc,
                             outc,
                             kernel_size=ks,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
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
        conv = nas_module.conv.conv
        cur_kernel = nas_module.conv.conv.kernel
        
        cur_kernel = cur_kernel.view(conv.ks[2], conv.ks[1], conv.ks[0], conv.inc, conv.outc)
        cur_kernel = cur_kernel[get_slice(conv.ks[2], self.ks[2]), 
                                get_slice(conv.ks[1], self.ks[1]), 
                                get_slice(conv.ks[0], self.ks[0]), :, :]
        cur_kernel = cur_kernel.contiguous().view(-1, conv.inc, conv.outc)

        if runtime_inc_constraint is not None:
            cur_kernel = cur_kernel[:, runtime_inc_constraint, :] if self.ks > (1,1,1) \
                else cur_kernel[runtime_inc_constraint]            
        else:
            cur_kernel = cur_kernel[:, torch.arange(self.inc), :] if self.ks > (1,1,1) \
                else cur_kernel[torch.arange(self.inc)]

        cur_kernel = cur_kernel[..., torch.arange(self.outc)]
        self.conv.conv.kernel.data = cur_kernel
        self.conv.bn.weight.data = nas_module.conv.bn.bn.weight[:self.outc]
        self.conv.bn.running_var.data = nas_module.conv.bn.bn.running_var[:self.outc]
        self.conv.bn.running_mean.data = nas_module.conv.bn.bn.running_mean[:self.outc]
        self.conv.bn.bias.data = nas_module.conv.bn.bn.bias[:self.outc]
        self.conv.bn.num_batches_tracked.data = \
            nas_module.conv.bn.bn.num_batches_tracked

    def forward(self, inputs):
        return self.conv(inputs)



class SPVConvolutionBlock(RandomModule):
    def __init__(self,
                 inc,
                 outc,
                 ks=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 depth=1,
                 bias=True,
                 no_bn=False,
                 no_relu=False,
                 transposed=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        if isinstance(ks, int):
            self.ks = (ks, ks, ks)
        self.bias = bias
        self.no_bn = no_bn
        self.no_relu = no_relu
        self.depth = depth

        self.conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 spnn.Conv3d(inc,
                             outc,
                             kernel_size=ks,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             transposed=transposed)),
                ('bn', spnn.BatchNorm(outc)),
                ('act',
                 spnn.ReLU(True) if not no_relu else nn.Sequential())
            ]))

        mlps = [LinearBlock(inc,
                            outc,
                            bias=bias,
                            no_relu=no_relu,
                            no_bn=no_bn)]
        for i in range(1, depth):
            mlps.append(
                LinearBlock(outc,
                            outc,
                            bias=bias,
                            no_relu=no_relu,
                            no_bn=no_bn))
        self.mlp = nn.Sequential(*mlps)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, nas_module, runtime_inc_constraint=None):
        conv = nas_module.conv.conv
        cur_kernel = nas_module.conv.conv.kernel
        
        cur_kernel = cur_kernel.view(conv.ks[2], conv.ks[1], conv.ks[0], conv.inc, conv.outc)
        cur_kernel = cur_kernel[get_slice(conv.ks[2], self.ks[2]), 
                                get_slice(conv.ks[1], self.ks[1]), 
                                get_slice(conv.ks[0], self.ks[0]), :, :]
        cur_kernel = cur_kernel.contiguous().view(-1, conv.inc, conv.outc)

        if runtime_inc_constraint is not None:
            cur_kernel = cur_kernel[:, runtime_inc_constraint, :] if self.ks > (1,1,1) \
                else cur_kernel[runtime_inc_constraint]            
        else:
            cur_kernel = cur_kernel[:, torch.arange(self.inc), :] if self.ks > (1,1,1) \
                else cur_kernel[torch.arange(self.inc)]

        cur_kernel = cur_kernel[..., torch.arange(self.outc)]
        self.conv.conv.kernel.data = cur_kernel
        self.conv.bn.weight.data = nas_module.conv.bn.bn.weight[:self.outc]
        self.conv.bn.running_var.data = nas_module.conv.bn.bn.running_var[:self.outc]
        self.conv.bn.running_mean.data = nas_module.conv.bn.bn.running_mean[:self.outc]
        self.conv.bn.bias.data = nas_module.conv.bn.bn.bias[:self.outc]
        self.conv.bn.num_batches_tracked.data = \
            nas_module.conv.bn.bn.num_batches_tracked
        
        mlps = nas_module.mlp
        for i, block in enumerate(self.mlp):
            if i == 0:            
                block.load_weights(mlps[i], runtime_inc_constraint)
            else:
                block.load_weights(mlps[i])

    def forward(self, inputs):
        p = voxel_to_point(inputs['voxel'],inputs['points'])
        p.F = self.mlp(p.F)
        out = self.conv(inputs['voxel'])
        out.F = out.F + point_to_voxel(out, p).F
        return out    










class DynamicSPVConvolutionBlock(RandomModule):
    # Search space : number of features, kernel, random depth for Linear
    # padding=None for automatic padding calculation based on kernel size.
    def __init__(self, 
                 inc, 
                 outc,
                 cr_bounds=(0.25, 1.0), 
                 possible_ks=((3,3,3),(5,5,5)),
                 depth_mlp=(1, 5),
                 stride=1,
                 padding = None,
                 bias=True,
                 no_bn=False,
                 no_relu=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.cr_bounds = cr_bounds
        self.possible_ks = possible_ks
        self.max_ks = tuple(max(sizes) for sizes in zip(*possible_ks))
        self.depth_min  = depth_mlp[0]
        self.depth_max  = depth_mlp[1]
        self.s = stride
        self.pad = padding
        self.bias = bias
        self.no_bn = no_bn
        self.no_relu = no_relu

        self.conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 SparseDynamicConv3dKer(inc,
                                        outc,
                                        kernel_size=self.max_ks,
                                        stride=stride,
                                        padding=padding)),
                ('bn', SparseDynamicBatchNorm(outc)),
                ('act',
                 spnn.ReLU(True) if not self.no_relu else nn.Sequential())
            ]))

        mlps = []
        linear = OrderedDict()
        linear['conv'] = DynamicLinear(inc, outc, bias=bias)
        if not self.no_bn:
            linear['bn'] = DynamicBatchNorm(outc, eps=1e-3, momentum=0.01)
        if not self.no_relu:
            linear['act'] = nn.ReLU(True)
        mlp = nn.Sequential(linear)
        mlps.append(mlp)
        for _ in range(1,self.depth_max):
            linear = OrderedDict()
            linear['conv'] = DynamicLinear(outc, outc, bias=bias)
            if not self.no_bn:
                linear['bn'] = DynamicBatchNorm(outc, eps=1e-3, momentum=0.01)
            if not self.no_relu:
                linear['act'] = nn.ReLU(True)
            mlp = nn.Sequential(linear)
            mlps.append(mlp)
        self.mlps = RandomDepth(
            *mlps,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
        )
        
        self.runtime_inc = None
        self.in_channel_constraint = None
        self.runtime_outc = None
        self.runtime_ks = None
        self.active_MLP = None
        self.runtime_depth = None

    def re_organize_middle_weights(self):
        weights = self.conv.conv.kernel.data
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
            self.conv.conv.kernel.data = torch.index_select(
                self.conv.conv.kernel.data, 2, sorted_idx)
        else:
            self.conv.conv.kernel.data = torch.index_select(
                self.conv.conv.kernel.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.conv.bn.bn, sorted_idx)

        for mlp in self.mlps.layers:
            weights = mlp.conv.linear.weight.data
            outc, inc = weights.shape
            importance = torch.sum(torch.abs(weights), dim=(1))

            sorted_importance, sorted_idx = torch.sort(importance,
                                                       dim=0,
                                                       descending=True)
            mlp.conv.linear.weight.data = torch.index_select(
                mlp.conv.linear.weight.data, 0, sorted_idx)
            if self.bias:
                mlp.conv.linear.bias.data = torch.index_select(
                    mlp.conv.linear.bias.data, 0, sorted_idx)
            adjust_bn_according_to_idx(mlp.bn.bn, sorted_idx)

    def constrain_in_channel(self, constraint):
        self.in_channel_constraint = constraint
        self.runtime_inc = None

    def manual_select_in(self, channel):
        if self.in_channel_constraint is not None:
            return
        self.runtime_inc = channel

    def manual_select(self, channel, ks, active_mlp, depth=None, padding=None):
        self.conv.conv.set_output_channel(channel)
        self.conv.bn.set_channel(channel)
        for i, mlp in enumerate(self.mlps.layers):
            mlp.conv.set_output_channel(channel)
            if not self.no_bn:
                mlp.bn.set_channel(channel)
            if i > 0:
                mlp.conv.set_in_channel(channel)
        self.runtime_outc = channel

        self.conv.conv.set_kernel_size(ks)
        self.runtime_ks = ks

        self.active_MLP = active_mlp
        self.mlps.manual_select(depth)
        self.runtime_depth = depth

        if padding is not None:
            self.conv.conv.set_padding(padding)
            self.pad = padding

    def random_sample(self):
        sample = {}

        cr = random.uniform(*self.cr_bounds)
        channel = make_divisible(int(cr * self.outc))
        sample['channel'] = channel
        ks = random.choice(self.possible_ks)
        sample['ks'] = ks
        active_mlp = random.choice([True, False])
        sample['active_mlp'] = active_mlp
        depth_min = 1 if self.depth_min is None else self.depth_min
        depth_max = len(self.mlps.layers) if self.depth_max is None else self.depth_max
        depth = random.randint(depth_min, depth_max)
        sample['depth'] = depth

        self.manual_select(channel, ks, active_mlp, depth)
        
        return sample

    def clear_sample(self):
        self.runtime_outc = None
        self.runtime_ks = None
        self.active_MLP = None
        self.runtime_depth = None

    def status(self):
        sample = {}
        sample['channel'] = self.runtime_outc
        sample['ks'] = self.runtime_ks
        sample['active_mlp'] = self.active_MLP
        sample['depth'] = self.runtime_depth
        return sample
    
    def determinize(self):
        if self.runtime_inc is None:
            assert self.in_channel_constraint is not None
            inc = len(self.in_channel_constraint)
        else:
            inc = self.runtime_inc

        if not self.active_MLP:
            determinized_model = ConvolutionBlock(inc,
                                                  self.runtime_outc,
                                                  self.runtime_ks,
                                                  self.s,
                                                  padding=self.pad,
                                                  no_relu=self.no_relu)
            determinized_model.load_weights(self, self.in_channel_constraint)
            return determinized_model
        else:
            determinized_model = SPVConvolutionBlock(inc,
                                                     self.runtime_outc,
                                                     ks=self.runtime_ks,
                                                     stride=self.s,
                                                     padding=self.pad,
                                                     depth=self.runtime_depth,
                                                     bias=self.bias,
                                                     no_bn=self.no_bn,
                                                     no_relu=self.no_relu)
            determinized_model.load_weights(self, self.in_channel_constraint)
            return determinized_model
    '''
    def forward(self, inputs):
        if self.in_channel_constraint is None:
            in_channel = inputs['voxel'].F.shape[-1]
            self.runtime_inc = in_channel
            self.net.conv.set_in_channel(in_channel=in_channel)
        else:
            self.net.conv.set_in_channel(constraint=self.in_channel_constraint)

        p = voxel_to_point(inputs['voxel'],inputs['points'])
        p.F = self.mlp(p.F)
        out = self.conv(inputs['voxel'])
        out.F = out.F + point_to_voxel(out, p).F
        return out    
    '''

