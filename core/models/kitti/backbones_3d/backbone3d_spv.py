from functools import partial

import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

import os, logging

from pcdet.models.backbones_3d.__init__ import __all__

from core.models.utils import *

def post_act_block_ts(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='tsconv', norm_fn=None):

    if conv_type == 'tsconv':
        conv = spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    elif conv_type == 'inverseconv':
        conv = spnn.Conv3d(in_channels, out_channels, kernel_size,  stride=stride, bias=False, transposed=True) 
    else:
        raise NotImplementedError

    m = nn.Sequential(
        conv,
        norm_fn(out_channels),
        spnn.ReLU(),
    )

    return m


class VoxelBackBone8xTSSPV(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(spnn.BatchNorm, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.voxel_size = kwargs.get('voxel_size', None)
        if self.voxel_size is None:
            raise ValueError('voxel_size is None')
        
        self.point_cloud_range = kwargs.get('point_cloud_range', None)
        if self.point_cloud_range is None:
            raise ValueError('point_cloud_range is None')

        self.conv_input = nn.Sequential(
            spnn.Conv3d(input_channels, 16, 3, padding=1, bias=False),
            spnn.BatchNorm(16),
            spnn.ReLU(),
        )

        block = post_act_block_ts

        self.conv1 = nn.Sequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = nn.Sequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='tsconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = nn.Sequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='tsconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = nn.Sequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='tsconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = nn.Sequential(
            # [200, 150, 5] -> [200, 150, 2]
            spnn.Conv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False),
            norm_fn(128),
            spnn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

        logging.warning('Built VoxelBackBone8x for TorchSparse')


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        # input_sp_tensor = spconv.SparseConvTensor(
        #     features=voxel_features,
        #     indices=voxel_coords.int(),
        #     spatial_shape=self.sparse_shape,
        #     batch_size=batch_size
        # )
        voxel_coords = voxel_coords.int()

        pc_features, pc_coords = batch_dict['points'][:, 1:], batch_dict["points"][:, :-1]
        voxel_size = torch.tensor([1] + self.voxel_size, device=pc_coords.device)
        lower_range = torch.tensor([0.0] + list(self.point_cloud_range[:3]), device=pc_coords.device)
        upper_range = torch.tensor([batch_size] + list(self.point_cloud_range[3:]), device=pc_coords.device)
        
        in_range_mask = (pc_coords >= lower_range) & (pc_coords <= upper_range)
        valid_mask = in_range_mask.all(dim=1)

        pc_coords = pc_coords[valid_mask]
        pc_features = pc_features[valid_mask] 
        pc_coords = (pc_coords - lower_range) / voxel_size

        # input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        spatial_range = (voxel_coords[:, 0].max().item() + 1,) + tuple(self.sparse_shape)
        input_sp_tensor = torchsparse.SparseTensor(voxel_features, voxel_coords, spatial_range=spatial_range)  # dimension match
        
        input_sp_tensor = PointTensor(input_sp_tensor.F, input_sp_tensor.C.float())
        print('batch_dict')
        print(batch_dict.keys())
        print()
        print('spatial_range')
        print(spatial_range)
        print(self.point_cloud_range)
        print()
        print('voxel_coords')
        print(torch.min(voxel_coords, 0)[0], torch.max(voxel_coords, 0)[0])
        print()
        print('pc_coords')
        print(torch.min(pc_coords, 0)[0], torch.max(pc_coords, 0)[0])
        print()
        print('voxel_features')
        print(torch.min(voxel_features, 0)[0], torch.max(voxel_features, 0)[0])
        print((torch.max(voxel_features, 0)[0]-lower_range)/torch.tensor([0.05, 0.05, 0.1, 1], device=voxel_features.device))
        print('pc_feature')
        print(torch.min(pc_features, 0)[0], torch.max(pc_features, 0)[0])
        print((torch.max(pc_features, 0)[0]-lower_range)/torch.tensor([0.05, 0.05, 0.1, 1], device=pc_features.device))
        print()
        print()

        '''
        point_1 = PointTensor(input_sp_tensor.F, input_sp_tensor.C.float())
        voxel_size = [0.05, 0.05, 0.1]
        voxel_size_tensor = torch.tensor(voxel_size, device=batch_dict["points"].device)
        coords = torch.round(batch_dict["points"][:, 1:-1] / voxel_size_tensor).int()
        coords -= coords.min(0, keepdim=True)[0]
        point_2 = PointTensor(batch_dict["points"][:, 1:], batch_dict["points"][:, :-1])
        print('spatial_range')
        print(self.sparse_shape,spatial_range)
        print((torch.tensor(range[3:])-torch.tensor(range[:3]))/torch.tensor([0.05, 0.05, 0.1]))
        print()
        print()
        '''
        '''

        print('point_1')
        print(torch.min(point_1.C, 0)[0], torch.max(point_1.C, 0)[0])
        print(point_1.C.shape,point_1.F.shape,point_1.s)
        print('point_range')
        print(torch.min(point_2.C, 0)[0], torch.max(point_2.C, 0)[0])
        print(point_2.C.shape,point_2.F.shape,point_2.s)
        print('coords')
        print(coords.min(0)[0],coords.max(0)[0])
        '''
        #voxel_2 = initial_voxelize(point_2, 1.0, voxel_size)


        #point_3 = PointTensor(input_sp_tensor.F, input_sp_tensor.C.float())
        #point_4 = PointTensor(input_sp_tensor.F, input_sp_tensor.C.float())

        #pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        #pc_ -= pc_.min(0, keepdims=1)
        #_, inds, inverse_map = sparse_quantize(pc_,
        #                                       return_index=True,
        #                                       return_inverse=True)
        #if 'train' in self.split:
        #    if len(inds) > self.num_points:
        #        inds = np.random.choice(inds, self.num_points, replace=False)
        #pc = pc_[inds]
        #feat = feat_[inds]

        '''
        # Base data
        print(batch_dict.keys())
        print()
        print('batch_dict["points"]')
        print(batch_dict['points'])
        print()
        print('batch_dict["voxel_features"]')
        print(batch_dict['voxel_features'])
        print()
        print('batch_dict["voxel_coords"]')
        print(batch_dict['voxel_coords'])
        print()
        print()

        # Initial pc and voxel
        print('spatial_range')
        print(spatial_range)
        print()
        print('input_sp_tensor')
        print(input_sp_tensor)
        print(input_sp_tensor.C)
        print(input_sp_tensor.F)
        print(input_sp_tensor.s)
        print()
        print('point')
        print(point)
        print(point.C)
        print(point.F)
        print(point.s)
        print()
        print()

        # Initial pc and voxel
        voxel_new = point_to_voxel(input_sp_tensor, point)
        point_new = voxel_to_point(input_sp_tensor, point)
        print('voxel_new')
        print(voxel_new)
        print(voxel_new.C)
        print(voxel_new.F)
        print(voxel_new.s)
        print()
        print('point_new')
        print(point_new)
        print(point_new.C)
        print(point_new.F)
        print(point_new.s)
        print()
        print()
        '''
        raise Exception('Debugging')

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

__all__['VoxelBackBone8xTSSPV'] = VoxelBackBone8xTSSPV