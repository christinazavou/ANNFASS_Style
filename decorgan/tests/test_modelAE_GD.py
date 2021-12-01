from unittest import TestCase
import numpy as np
import torch

from modelAE_GD import style_encoder_8, style_encoder_16, style_encoder_32, discriminator, generator_halfsize_x8_adain, \
    common_discriminator_1, discriminator_part_global_plausibility_1, discriminator_part_style_plausibility_1


# from modelAE_GD_new import discriminator_2crop14kernel, discriminator_2crop10kernel


class Teststyle_encoder_generic(TestCase):

    def test_style_8(self):
        se = style_encoder_8()
        x = np.random.random([1, 1, 100, 120, 98])
        x = torch.from_numpy(x).float()
        out = se.conv_0(x)
        print(out.shape)
        out = se.conv_1(out)
        print(out.shape)

    def test_generic_on_cuda(self):
        se = style_encoder_16()
        se.to(torch.device('cuda'))

        x = np.random.random([1, 1, 10, 20, 30])
        x = torch.from_numpy(x).float().to(torch.device('cuda'))

        r = se(x)
        print(r)

    def test_16(self):
        x = np.random.random([1, 1, 10, 20, 30])
        x = torch.from_numpy(x).float()
        y = np.random.random([1, 16])
        y = torch.from_numpy(y).float()

        se = style_encoder_16()
        r = se(x)
        print(r.shape)
        loss = torch.mean(y - r)
        loss.backward()

        se = style_encoder_16(pool_method='avg')
        r = se(x)
        print(r.shape)
        loss = torch.mean(y - r)
        loss.backward()

        se = style_encoder_16(kernel=3)
        r = se(x)
        print(r.shape)
        loss = torch.mean(y - r)
        loss.backward()

        se = style_encoder_16(dilation=False)
        r = se(x)
        print(r.shape)
        loss = torch.mean(y - r)
        loss.backward()

    def test_32(self):
        x = np.random.random([1, 1, 100, 120, 98])
        x = torch.from_numpy(x).float()
        se = style_encoder_32()
        r = se(x)

        x = np.random.random([1, 1, 10, 20, 30])
        x = torch.from_numpy(x).float()
        y = np.random.random([1, 32])
        y = torch.from_numpy(y).float()

        se = style_encoder_32()
        r = se(x)
        print(r.shape)
        loss = torch.mean(y - r)
        loss.backward()

        se = style_encoder_32(pool_method='avg')
        r = se(x)
        print(r.shape)
        loss = torch.mean(y - r)
        loss.backward()

        se = style_encoder_32(kernel=3)
        r = se(x)
        print(r.shape)
        loss = torch.mean(y - r)
        loss.backward()

        se = style_encoder_32(dilation=False)
        r = se(x)
        print(r.shape)
        loss = torch.mean(y - r)
        loss.backward()


import torch.nn.functional as F


class Testdiscriminator(TestCase):

    def get_style_voxel_Dmask_multivox_tensor(self, vox):
        # 256 -crop- 252 -maxpoolk14s2- 120
        crop_margin = 2
        kernel_size = 14
        vox_tensor = torch.from_numpy(
            vox[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]).to(
            torch.device('cuda')).float()
        smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel_size, stride=2, padding=0)
        smallmask = torch.round(smallmask_tensor).type(torch.uint8)
        return smallmask

    def test_weight_init(self):
        torch.manual_seed(10)
        discr = discriminator(32, 17, init_weights=True)
        print(torch.sum(discr.conv_1.weight))
        discr = discriminator(32, 17, init_weights=False)
        print(torch.sum(discr.conv_1.weight))
        discr = discriminator(32, 17, init_weights=True)
        print(torch.sum(discr.conv_1.weight))
        discr = discriminator(32, 17, init_weights=False)
        print(torch.sum(discr.conv_1.weight))

    def test_weight_init2(self):
        discr = style_encoder_32(init_weights=True)
        print(torch.sum(discr.convolutions[0].weight))
        discr = style_encoder_32(init_weights=False)
        print(torch.sum(discr.convolutions[0].weight))

    def test_forward(self):
        discr = discriminator(32, 17)
        inp = np.random.random((1, 1, 80, 160, 120))
        inp = torch.from_numpy(inp).float()
        print(discr(inp).shape)
        return

        discr = discriminator(2, 17)
        inp = np.random.random((4, 1, 64, 64, 64))
        print(self.get_style_voxel_Dmask_multivox_tensor(inp).shape)
        inp = torch.from_numpy(inp).float()
        print(discr(inp).shape)

        inp = np.random.random((4, 1, 128, 128, 128))
        print(self.get_style_voxel_Dmask_multivox_tensor(inp).shape)
        inp = torch.from_numpy(inp).float()
        print(discr(inp).shape)

        inp = np.random.random((4, 1, 32, 32, 32))
        print(self.get_style_voxel_Dmask_multivox_tensor(inp).shape)
        inp = torch.from_numpy(inp).float()
        print(discr(inp).shape)

    def test_batch_gradient1(self):
        discr = discriminator(16, 16)

        inp = np.random.random((1, 1, 100, 120, 98))
        inp = torch.from_numpy(inp).float()

        targ = np.ones((1, 16, 42, 52, 41))
        targ = torch.from_numpy(targ).float()
        outp = discr(inp)

        print(discr.conv_1.weight.grad)
        loss = (targ - outp).pow(2).sum()
        loss.backward()
        print(discr.conv_1.weight.grad[:, 0,0,0,0])

        optim = torch.optim.SGD(discr.parameters(), lr=0.1)
        print(discr.conv_1.weight[:,0,0,0,0])
        optim.step()
        print(discr.conv_1.weight[:,0,0,0,0])

    def test_batch_gradient2(self):
        x = torch.rand((10, 20))
        a = torch.rand((20), requires_grad=True)
        y = (x+a)**2
        targ = torch.ones((10, 20))
        loss = (targ - y).pow(2).sum()
        print(a.grad)
        loss.backward()
        print(a.grad)
        optim = torch.optim.Adam([a], lr=0.01)
        print(a)
        optim.step()
        print(a)
        print("-----------------")

        a = torch.rand((20), requires_grad=True)
        optim = torch.optim.Adam([a], lr=0.01)
        for i in range(10):
            x = torch.rand((1, 20))
            y = (x+a)**2
            targ = torch.ones((1, 20))
            loss = (targ - y).pow(2).sum()
            loss /= 10
            loss.backward()
        print(a.grad)
        print(a)
        optim.step()
        print(a)


class Testdiscriminator_2crop14kernel(TestCase):

    def test_me(self):
        inp = torch.rand((1, 1, 40, 60, 30))

        discr = discriminator_2crop14kernel(32, 17)
        print(discr.get_crop())
        print(discr.get_pool_kernel())
        print(discr(inp).shape)
        print(discr.agrees_mask_shape(inp.shape))

        discr = discriminator_2crop10kernel(32, 17)
        print(discr.get_crop())
        print(discr.get_pool_kernel())
        print(discr(inp).shape)
        print(discr.agrees_mask_shape(inp.shape))


class Testgenerator_halfsize_x8_adain(TestCase):
    def test_me(self):
        content = torch.rand((1, 1, 10, 20, 15))
        mask = torch.rand((1, 1, 20, 40, 30))
        style = torch.rand((1, 1, 40, 80, 60))
        gen = generator_halfsize_x8_adain(16, )
        res = gen(content, style, mask)
        print(res.shape)


class TestDiscriminator(TestCase):
    def test_patches(self):
        voxel = torch.rand((1, 1, 100, 60, 80))
        d = discriminator(16, 8)
        result = d(voxel)
        print(f"D_out: {result.shape}")
        result = d.layer(voxel, layer='last')[:, 0:-1].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        print(f"D_out_style: {result.shape}")
        result = d(voxel)
        result = torch.nn.AdaptiveMaxPool3d(1)(result)[:, 0:-1, :, :, :]
        print(f"D_out_style: {result.shape}")

        result = d(voxel)
        from utils.patch_generator import get_random_non_cube_patches
        patches = get_random_non_cube_patches(result[0, 0:-1], patch_factor=2, patch_num=8, stride_factor=1)
        print(f"D_out patches: {patches.shape}")
        patches = get_random_non_cube_patches(result[0, 0:-1], patch_factor=4, patch_num=64, stride_factor=1)
        print(f"D_out patches: {patches.shape}")
        patches = get_random_non_cube_patches(result[0, 0:-1], patch_factor=2, patch_num=27, stride_factor=2)
        print(f"D_out patches: {patches.shape}")

        patches_embeddings = d.pool(patches)
        print(f"patches_embeddings: {patches_embeddings.shape}")


class TestAny(TestCase):
    def test_shapes(self):
        voxel = torch.rand((1, 1, 100, 120, 98))
        discriminator_common_part = common_discriminator_1(16,)
        discriminator_global_part = discriminator_part_global_plausibility_1(16)
        discriminator_style_part = discriminator_part_style_plausibility_1(16, 128)
        d_common = discriminator_common_part(voxel)
        print(f"d_common: {d_common.shape}")
        d_global = discriminator_global_part(d_common)
        print(f"d_global: {d_global.shape}")
        d_style = discriminator_style_part(d_common)
        print(f"d_style: {d_style.shape}")

    def test_pool(self):
        voxel = torch.rand((1, 32, 100, 120, 98))
        pool = torch.nn.AdaptiveMaxPool3d(1)
        out = pool(voxel)[:, :, 0, 0, 0].numpy()
        out2 = np.max(voxel.numpy(), (2, 3, 4))
        assert np.array_equal(out, out2)
