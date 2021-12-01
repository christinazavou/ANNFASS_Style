from unittest import TestCase
import torch

from modelAE_GD import discriminator
from utils.patch_gen_2 import get_random_paired_non_cube_patches
import numpy as np
import torch.nn.functional as F


def get_patches_style_voxel_Dmask_tensor(vox):
    # 256 -crop- 252 -maxpoolk14s2- 120
    crop_margin = 2
    kernel_size = 14
    vox_tensor = vox[:, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
    vox_tensor = vox_tensor.unsqueeze(1).float()
    print(vox_tensor.shape)
    smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel_size, stride=2, padding=0)
    print(smallmask_tensor.shape)
    smallmask = torch.round(smallmask_tensor).type(torch.uint8)
    return smallmask


def get_patches_content_voxel_Dmask_tensor(vox):
    upsample_rate = 8
    crop_margin = 1
    vox_tensor = vox.unsqueeze(1).float()
    # input
    smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=upsample_rate, stride=upsample_rate, padding=0)
    # Dmask
    smallmask_tensor = smallmaskx_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin,
                                         crop_margin:-crop_margin]
    smallmask_tensor = F.interpolate(smallmask_tensor, scale_factor=upsample_rate // 2, mode='nearest')
    smallmask_tensor = torch.round(smallmask_tensor).type(torch.uint8)
    return smallmask_tensor


class Test(TestCase):
    def test_get_random_paired_non_cube_patches(self):
        content_vox = torch.zeros((10,20,30))
        generated_vox = torch.zeros((10,20,30))

        content_patches, generated_patches, xyz_patches = get_random_paired_non_cube_patches(
            content_vox,generated_vox)
        print(content_patches.shape)
        print(np.any(np.array(content_patches.shape) == 0))

    def test_get_patches_style_voxel_Dmask_tensor(self):
        # vox = torch.rand((16,20,32,20))
        vox = torch.rand((16,24,28,24))

        print((np.array(vox.shape[1:]) - 4 - 14) / 2 + 1)

        smallmask = get_patches_style_voxel_Dmask_tensor(vox)
        print(smallmask.shape)
        smallmask = get_patches_content_voxel_Dmask_tensor(vox)
        print(smallmask.shape)

    def test_get_patches_content_voxel_Dmask_tensor(self):
        vox = torch.rand((16,20,32,20))
        print(np.any(np.array(vox.shape[1:]) // 8 <= 2))
        vox = torch.rand((16,24,34,24))
        print(np.any(np.array(vox.shape[1:]) // 8 <= 2))
        # smallmask = get_patches_content_voxel_Dmask_tensor(vox)
        # print(smallmask.shape)

    def test_discriminator(self):
        D = discriminator(32, 17)
        voxel_fake_patches = torch.rand((16,1,20,32,20))
        res = D(voxel_fake_patches)
        print(res.shape)

    def test_eq(self):
        in_shape = np.array([20,32,20])
        out_1 = ((((in_shape-4/2+1 -3)/2+1-3)/1+1-3)/1+1-3)/1+1
        out_2 = (in_shape-4-12)/2+1
        print(out_1, out_2)


if __name__ == '__main__':
    Test().test_get_patches_style_voxel_Dmask_tensor()
