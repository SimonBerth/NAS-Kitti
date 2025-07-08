import torchsparse.nn as spnn

test = spnn.Conv3d(64, 64, kernel_size=3, stride=2, padding=(0, 1, 1),)
