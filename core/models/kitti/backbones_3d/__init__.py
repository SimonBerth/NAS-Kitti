#TODO: Remove unused backbones
from .backbone3d import VoxelBackBone8xTS
from .backbone3d_spv import VoxelBackBone8xTSSPV
from .backbone3d_spv_alter import VoxelBackBone8xTSSPV_Alter
from .unet import UNetV2TS
from .backbone_voxel_next import VoxelResBackBone8xVoxelNeXtTS
from .backbone3d_nas import VoxelBackBone8xTSNAS

__all__ = {
    'VoxelBackBone8xTS': VoxelBackBone8xTS,
    'UNetV2TS': UNetV2TS,
    'VoxelResBackBone8xVoxelNeXtTS': VoxelResBackBone8xVoxelNeXtTS,
    'VoxelBackBone8xTSSPV': VoxelBackBone8xTSSPV,
    'VoxelBackBone8xTSSPV_Alter': VoxelBackBone8xTSSPV_Alter,
    'VoxelBackBone8xTSSPV_NAS': VoxelBackBone8xTSNAS,
}