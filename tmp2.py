import argparse
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

from core.modules.new_layers import DynamicSPVConvolutionBlock

class RandomDataset:
    def __init__(self, input_size: int, voxel_size: float) -> None:
        self.input_size = input_size
        self.voxel_size = voxel_size

    def __getitem__(self, _: int) -> Dict[str, Any]:
        inputs = np.random.uniform(-100, 100, size=(self.input_size, 4))
        labels = np.random.choice(10, size=self.input_size)

        coords, feats = inputs[:, :3], inputs
        coords -= np.min(coords, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords, self.voxel_size, return_index=True)

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        labels = torch.tensor(labels[indices], dtype=torch.long)

        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)
        return {"input": input, "label": label}

    def __len__(self):
        return 100





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_enabled", action="store_true")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = RandomDataset(input_size=10000, voxel_size=0.2)
    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=sparse_collate_fn,
    )

    model = DynamicSPVConvolutionBlock(
        4, 
        128,
        cr_bounds=(0.125, 1.0), 
        possible_ks=((3,3,3),(5,5,5)),
        depth_mlp=(1, 5),
        stride=1, 
        bias=True,
        no_bn=False,
        no_relu=False
    ).to(args.device)

    print(f"Model: {model.__repr__()}")

    model.re_organize_middle_weights()
    model.manual_select_in(3)
    model.constrain_in_channel(4)
    model.manual_select(channel=16, ks=(3, 3, 3), active_mlp=True, depth=2)

    channel_sizes = []
    for i in range(1000):
        sample = model.random_sample()
        channel_sizes.append(sample["channel"])
    
    from collections import Counter
    channel_counter = Counter(channel_sizes)
    for size in sorted(channel_counter.keys()):
        print(f"Channel size {size}: {channel_counter[size]} samples")

'''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler(enabled=args.amp_enabled)

    for k, feed_dict in enumerate(dataflow):
        inputs = feed_dict["input"].to(device=args.device)
        labels = feed_dict["label"].to(device=args.device)

        with amp.autocast(enabled=args.amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs.feats, labels.feats)

        print(f"[step {k + 1}] loss = {loss.item()}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # enable torchsparse 2.0 inference
    model.eval()
    # enable fused and locality-aware memory access optimization
    torchsparse.backends.benchmark = True  # type: ignore

    with torch.no_grad():
        for k, feed_dict in enumerate(dataflow):
            inputs = feed_dict["input"].to(device=args.device).half()
            labels = feed_dict["label"].to(device=args.device)

            with amp.autocast(enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs.feats, labels.feats)

            print(f"[inference step {k + 1}] loss = {loss.item()}")
'''

'''  
test = RandomDataset(10,1.0)

Conv = SparseDynamicConv3d(4, 4, kernel_size=5, stride=1)
Conv.set_in_channel(4)
Conv.set_output_channel(4)
#Conv.set_kernel_size(3)

net = nn.Sequential(OrderedDict([('conv',Conv),('act', spnn.ReLU(True))])).to("cuda")

x = test[0]['input'].to("cuda")

print(x)
print("F dtype:", x.F.dtype, "device:", x.F.device)
print("C dtype:", x.C.dtype, "device:", x.C.device)
print("Shapes:", x.F.shape, x.C.shape)
print(x.F)
print(x.C)


y = net(x)


print(y)
print("F dtype:", x.F.dtype, "device:", x.F.device)
print("C dtype:", x.C.dtype, "device:", x.C.device)
print(y.F.is_cuda, y.F.requires_grad, y.F.dtype)
print(torch.isnan(y.F).any(), torch.isinf(y.F).any())
print("Shapes:", x.F.shape, x.C.shape)
print(y.F)
print(y.C)
'''