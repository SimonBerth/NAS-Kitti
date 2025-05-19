#TODO : Refactor KITTIInternal.__getitem__ (et al.) using super().__getitem__ to pass tensors to torchsparse
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.config import cfg, cfg_from_yaml_file
from torchsparse.utils.collate import sparse_collate_fn

__all__ = ['KITTI']


cfg_from_yaml_file('configs/kitti/default.yaml', cfg)


class KITTI(dict):

    def __init__(self, **kwargs):
        submit_to_server = kwargs.get('submit', False)

        if submit_to_server:
            super().__init__({
                'train':
                    KITTIInternal(split='train',
                                  submit=True),
                'test':
                    KITTIInternal(split='test',
                                  submit=True)
            })
        else:
            super().__init__({
                'train':
                    KITTIInternal(split='train'),
                'test':
                    KITTIInternal(split='test')
            })


class KITTIInternal(KittiDataset):

    def __init__(self,
                 split,
                 submit=False):
        
        if submit:
            cfg.DATA_CONFIG.DATA_SPLIT = {'train': 'trainval', 'test': 'test'}
            cfg.DATA_CONFIG.INFO_PATH = {'train': ['kitti_infos_trainval.pkl'], 'test': ['kitti_infos_test.pkl']}
        if split == 'train':
            training = True
        elif split == 'test':
            training = False
        else:
            raise ValueError(f"Split must be train of test, instead recieved : {split}")


        super().__init__(
            dataset_cfg = cfg.DATA_CONFIG, 
            class_names = cfg.CLASS_NAMES, 
            training=training, 
            root_path=None, 
            logger=None,
        )
    
    #TODO : Possibly change to Opcdet's dataset.collate_batch (check https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/__init__.py)
    #TODO : Modify back if we change code for torchsparse
    @staticmethod
    def collate_fn(inputs):
        return KittiDataset.collate_batch(inputs)
    #    return sparse_collate_fn(inputs)
    
