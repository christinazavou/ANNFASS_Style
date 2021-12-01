import torch
from torch.nn import functional as F
import numpy as np

image = torch.randn(1, 1, 60, 60)
coor = torch.LongTensor(60, 60, 2, 8)
coor.fill_(0) # Just to get valid indices

kc, kh, kw = 1, 3, 3  # kernel size
dc, dh, dw = 1, 1, 1  # stride
pad = (1,1,1,1)


def original(image):

    paddedimage = F.pad(image, pad)
    patches = paddedimage.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(-1, kc, kh, kw)

    a = 0
    for i in range(60):
        for j in range(60):
            coordinate = [i,j]
            indices = np.matrix(coor[(coordinate[0],coordinate[1])])
            patches[a,0,2,1] = image[0,0,indices[1,0],indices[0,0]]
            patches[a,0,2,2] = image[0,0,indices[1,1],indices[0,1]]
            patches[a,0,1,2] = image[0,0,indices[1,2],indices[0,2]]
            patches[a,0,0,2] = image[0,0,indices[1,3],indices[0,3]]
            patches[a,0,0,1] = image[0,0,indices[1,4],indices[0,4]]
            patches[a,0,0,0] = image[0,0,indices[1,5],indices[0,5]]
            patches[a,0,1,0] = image[0,0,indices[1,6],indices[0,6]]
            patches[a,0,2,0] = image[0,0,indices[1,7],indices[0,7]]
            a += 1
    # Reshape back
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(1, output_c, output_h, output_w)

    return patches_orig


def new(image):
    paddedimage = F.pad(image, pad)
    patches = paddedimage.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    patches = patches.contiguous()

    coor_in = coor.unsqueeze(0).unsqueeze(0)
    patches[0, 0, :, :,0,2,1] = image[0,0,coor_in[0, 0, :, :, 1,0],coor_in[0, 0, :,:,0,0]]
    patches[0, 0, :, :,0,2,2] = image[0,0,coor_in[0, 0, :, :, 1,1],coor_in[0, 0, :,:,0,1]]
    patches[0, 0, :, :,0,1,2] = image[0,0,coor_in[0, 0, :, :, 1,2],coor_in[0, 0, :,:,0,2]]
    patches[0, 0, :, :,0,0,2] = image[0,0,coor_in[0, 0, :, :, 1,3],coor_in[0, 0, :,:,0,3]]
    patches[0, 0, :, :,0,0,1] = image[0,0,coor_in[0, 0, :, :, 1,4],coor_in[0, 0, :,:,0,4]]
    patches[0, 0, :, :,0,0,0] = image[0,0,coor_in[0, 0, :, :, 1,5],coor_in[0, 0, :,:,0,5]]
    patches[0, 0, :, :,0,1,0] = image[0,0,coor_in[0, 0, :, :, 1,6],coor_in[0, 0, :,:,0,6]]
    patches[0, 0, :, :,0,2,0] = image[0,0,coor_in[0, 0, :, :, 1,7],coor_in[0, 0, :,:,0,7]]

    # Reshape back
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(1, output_c, output_h, output_w)

    return patches_orig


# print((original(image.clone()) - new(image.clone())).abs().max())

# S = 128 # channel dim
# W = 256 # width
# H = 256 # height
# batch_size = 10
# x = torch.randn(batch_size, S, W, H)
# size = 64 # patch size
# stride = 64 # patch stride
# print(x.shape)
# print(x.unfold(1, size, stride).shape)
# print(x.unfold(1, size, stride).unfold(2, size, stride).shape)
# print(x.unfold(1, size, stride).unfold(2, size, stride).unfold(3, size, stride).shape)
#
# stride = 16 # patch stride
# print(x.shape)
# print(x.unfold(1, size, stride).shape)
# print(x.unfold(1, size, stride).unfold(2, size, stride).shape)
# print(x.unfold(1, size, stride).unfold(2, size, stride).unfold(3, size, stride).shape)


V = 256  # voxel dim
batch_size = 4
x = torch.randn(batch_size, 1, V, V, V)
size = 32  # patch size
stride = 16  # patch stride
print(x.shape)
print(x.unfold(2, size, stride).shape)
print(x.unfold(2, size, stride).unfold(3, size, stride).shape)
print(x.unfold(2, size, stride).unfold(3, size, stride).unfold(4, size, stride).shape)
patches = x.unfold(2, size, stride).unfold(3, size, stride).unfold(4, size, stride).contiguous().view(batch_size, -1, size, size, size)
print(patches.shape)

import random
indices = random.sample(range(patches.shape[1]), 16)
indices = torch.tensor(indices)
sampled_patches = patches[:, indices]
print(sampled_patches.shape)
