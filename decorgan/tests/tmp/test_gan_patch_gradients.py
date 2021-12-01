import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from modelAE_GD import discriminator, generator_halfsize_x8_allstyles, style_encoder_8
from runners.common import plot_grad_flow
from utils.patch_gen import get_random_paired_patches as get_random_paired_patches_1
from utils.patch_gen_2 import get_random_paired_patches as get_random_paired_patches_2


def get_patches_content_voxel_Dmask(vox_tensor):
    device = torch.device('cuda')
    upsample_rate = 8
    crop_margin = 1
    # input
    smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=upsample_rate, stride=upsample_rate, padding=0)
    # Dmask
    smallmask_tensor = smallmaskx_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin,
                       crop_margin:-crop_margin]
    smallmask_tensor = F.interpolate(smallmask_tensor, scale_factor=upsample_rate // 2, mode='nearest')
    smallmask_tensor = torch.round(smallmask_tensor).type(torch.uint8)
    return smallmask_tensor


def run():
    x = torch.from_numpy(np.random.random((80,112,128))).cuda().float()

    D = discriminator(8, 8).cuda()
    G = generator_halfsize_x8_allstyles(8, 8).cuda()
    SE = style_encoder_8().cuda()

    SE.zero_grad()
    G.zero_grad()

    xs = torch.from_numpy(np.random.random((10,14,16))).cuda().unsqueeze(0).unsqueeze(0).float()
    z_tensor = SE(x.unsqueeze(0).unsqueeze(0))
    # z_tensor = torch.from_numpy(np.random.random((1, 8, 1, 1, 1))).cuda().float()
    mask_tensor = torch.from_numpy(np.random.randint(0, 2, size=(1,1,20,28,32))).cuda().float()
    gx = G(xs, z_tensor, mask_tensor)
    # xpatches, gxpatches, xyzpatches = get_random_paired_patches_1(x, gx[0,0])
    xpatches, gxpatches, xyzpatches = get_random_paired_patches_2(x, gx[0,0])
    dgxpatches = D(gxpatches.unsqueeze(1).float())

    Dmask_fake_patches = get_patches_content_voxel_Dmask(xpatches.unsqueeze(1).float())

    exception_raised = False
    try:
        print("ge before:", G.conv_1.weight.grad.sum())
    except:
        exception_raised = True
    assert exception_raised

    exception_raised = False
    try:
        print("se before:", SE.conv_1.weight.grad.sum())
    except:
        exception_raised = True
    assert exception_raised

    exception_raised = False
    try:
        print("d before:", D.conv_1.weight.grad.sum())
    except:
        exception_raised = True
    assert exception_raised

    loss_g = (torch.sum(
        (dgxpatches[:, 1:1 + 1] - 1) ** 2 * Dmask_fake_patches) +
              torch.sum((dgxpatches[:, -1:] - 1) ** 2 * Dmask_fake_patches)) / torch.sum(Dmask_fake_patches)
    loss_g.backward()
    print("ge after:", G.conv_1.weight.grad.sum())
    print("se after:", SE.conv_1.weight.grad.sum())
    print("d after:", D.conv_1.weight.grad.sum())
    plot_grad_flow(G.named_parameters(), title="Generator")
    plt.show()
    plot_grad_flow(SE.named_parameters(), title="Style Encoder")
    plt.show()


if __name__ == '__main__':
    run()
