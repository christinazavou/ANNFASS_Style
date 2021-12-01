import torch
import torch.nn.functional as F

import numpy as np


def expected_shape(in_shape, kernel, stride, padding):
    vox_tensor = torch.rand(in_shape)
    vox_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel, stride=stride, padding=padding)
    return vox_tensor.shape
print(expected_shape((1, 1, 110, 88, 100), 4, 1, 0))
print(expected_shape(expected_shape((1, 1, 110, 88, 100), 4, 1, 0), 3, 2, 0))
print(expected_shape((1, 1, 40, 70, 60), 8, 8, 0))


def expected_shape(in_shape, kernel, stride, padding):
    return np.floor((np.array(in_shape) - kernel + 2 * padding) / stride + 1)
print(expected_shape((40, 70, 60), 8, 8, 0))


def expected_shape(in_shape, kernel, stride, padding, crop, scale):
    vox_tensor = torch.rand(in_shape)
    vox_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel, stride=stride, padding=padding)
    vox_tensor = vox_tensor[:, :, crop:-crop, crop:-crop, crop:-crop]
    vox_tensor = F.interpolate(vox_tensor, scale_factor=scale, mode='nearest')
    return vox_tensor.shape
# print(expected_shape((1, 1, 40, 70, 60), 8, 8, 0, 1, 4))
print(expected_shape((1, 1, 40, 48, 40), 8, 8, 0, 1, 4))


