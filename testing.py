import argparse
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn as nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner, MaxSaver, Saver
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.callbacks import MeanIoU
#from core.trainers import KITTITrainer

import torchsparse.nn.functional as F

from pcdet.models import model_fn_decorator

import torchsparse.backends
torchsparse.backends.hash_rsv_ratio=4.0




F.set_conv_mode(2)

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.distributed:
        dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2 ** 32 - 1)

    seed = configs.train.seed + dist.rank(
    ) * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = builder.make_dataset()
    dataflow = {}
    sampler = {}
    for split in dataset:
        #TODO : Check if sampler and dataflow are working properly for Kitti
        sampler[split] = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler[split],
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    model = builder.make_model().cuda()

    #TODO: make_criterion() does nothing for Kitti
    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)

    # load checkpoint if it is possible
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=configs.distributed, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        start_epoch = it = 0
        last_epoch = -1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if configs.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.local_rank()% torch.cuda.device_count()], find_unused_parameters=True)
    
    scheduler = builder.make_scheduler(optimizer)
'''
    from pathlib import Path
    from tensorboardX import SummaryWriter
    root_dir = Path(__file__).resolve().parent 
    EXP_GROUP_PATH = '/'.join(args.config.split('/')[1:-1])
    TAG = Path(args.config).stem
    output_dir = root_dir / 'output' / EXP_GROUP_PATH / TAG 
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if dist.local_rank() == 0 else None

    from core.modules.train_utils import train_model
    train_model(
        model,
        optimizer,
        dataflow['train'],
        model_func=model_fn_decorator(),
        lr_scheduler=scheduler,
        optim_cfg=configs.optimizer,
        start_epoch=start_epoch,
        total_epochs=configs.num_epochs,
        start_iter=it,
        rank=dist.local_rank(),
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=sampler['train'],
        lr_warmup_scheduler=None,
        ckpt_save_interval=configs.ckpt_save_interval,
        max_ckpt_save_num=configs.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=False, 
        logger=logger, 
        logger_iter_interval=50,
        ckpt_save_time_interval=300,
        use_logger_to_record=True, 
        show_gpu_stat=True,
        use_amp=False,
        cfg=configs
    )
'''

if __name__ == '__main__':
    main()
