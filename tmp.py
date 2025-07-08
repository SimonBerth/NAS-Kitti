from core.modules.new_dynamic_sparseop import SparseDynamicConv3dKer
import torch
import torchsparse

ker = (5, 7, 9)  # Example kernel size
run_ker = (3, 5, 7)  # Example runtime kernel size

Conv = SparseDynamicConv3dKer(1, 1, kernel_size=ker, stride=1).to("cuda")
Conv.set_in_channel(1)
Conv.set_output_channel(1)
Conv.set_kernel_size(run_ker)
print(Conv.__repr__())

# Example: 4 points, 3D input, batch size 1
coords = torch.tensor([
    [0, 11, 11, 11], # z=-1, y=-1, x=-1
    [0, 9 , 11, 11], # z= 1, y=-1, x=-1
    [0, 11, 9 , 11], # z=-1, y= 1, x=-1
    [0, 9 , 9 , 11], # z= 1, y= 1, x=-1
    [0, 10, 10, 10], # z= 0, y= 0, x= 0
    [0, 11, 11, 9 ], # z=-1, y=-1, x= 1
    [0, 9 , 11, 9 ], # z= 1, y=-1, x= 1
    [0, 11, 9 , 9 ], # z=-1, y= 1, x= 1
    [0, 9 , 9 , 9 ], # z= 1, y= 1, x= 1
], dtype=torch.int32).cuda()

'''
feats = torch.tensor([
    [1.0, 1.0, 1.0, 1.0,],
    [1.0, 1.0, 1.0, 1.0,],
    [1.0, 1.0, 1.0, 1.0,],
    [1.0, 1.0, 1.0, 1.0,],
    [1.0, 1.0, 1.0, 1.0,],
],dtype=torch.float32).cuda()  # 4 points, 16 input feature channels
'''
feats = torch.tensor([
    [0.0,],
    [0.0,],
    [0.0,],
    [0.0,],
    [1.0,],
    [0.0,],
    [0.0,],
    [0.0,],
    [0.0,],
],dtype=torch.float32).cuda()  # 4 points, 16 input feature channels

# Create SparseTensor
x = torchsparse.SparseTensor(
    feats=feats,
    coords=coords
).to("cuda")

#print(x)
#print("F dtype:", x.F.dtype, "device:", x.F.device)
#print("C dtype:", x.C.dtype, "device:", x.C.device)
#print("Shapes:", x.F.shape, x.C.shape)
#print(x.F)
#print(x.C)

y = Conv(x)

print(y)
print("F dtype:", x.F.dtype, "device:", x.F.device)
print("C dtype:", x.C.dtype, "device:", x.C.device)
print(y.F.is_cuda, y.F.requires_grad, y.F.dtype)
print(torch.isnan(y.F).any(), torch.isinf(y.F).any())
print("Shapes:", x.F.shape, x.C.shape)
print(y.F)
print(y.C)