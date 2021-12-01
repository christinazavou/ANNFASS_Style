from unittest import TestCase
import torch
import sys
import numpy as np
import os
import logging
from MinkowskiEngine import SparseTensor

from lib.solvers import initialize_optimizer
from lib.utils import get_with_component_criterion
from config import get_config
from models.hrnet import HRNet3S2BD256
from models.hrnet_style_cls import HRNetStyleCls3S2BND128, HRNetStyleCls3S2BD256
from models.res16unet_style_cls import StyleRes16UNet34A

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


testargs = ["python test_resunet_style_cls.py",
            "--stylenet_path", "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_reconstruction_splits/ply_10K/split_train_val_test_debug",
            "--dataset", "StylenetXYZAEVoxelization0_01Dataset",
            "--input_feat", "coords",
            "--chamfer_loss", "True",
            "--bn_momentum", "0.02",
            "--export_feat", "False",
            "--conv1_kernel_size", "5"]
with patch.object(sys, 'argv', testargs):
    config = get_config()


# DEVICE = 'cuda'
DEVICE = 'cpu'


class TestHrnetStyleCls(TestCase):

    def make_sample_data(self, in_channels, out_channels):
        samples = 100
        components_cnt = 8

        p_coords = np.random.randint(-50, 50, (samples, 4))
        p_coords = torch.tensor(p_coords).int()
        p_input = np.random.random((samples, in_channels))
        p_input = torch.tensor(p_input).float()
        sinput = SparseTensor(p_input, p_coords).to(DEVICE)

        c_coords = np.random.randint(-50, 50, (components_cnt, 4))
        c_coords = torch.tensor(c_coords).int()

        c_indices = np.random.randint(0, 2, (8, samples))
        assert not any(np.sum(c_indices, 1) == 0)
        c_indices = torch.tensor(c_indices).float()
        sc_indices = SparseTensor(c_indices, c_coords).to(DEVICE)

        c_targets = np.random.randint(0, 8, (8, 1))
        c_targets = torch.tensor(c_targets).float()
        sc_targets = SparseTensor(c_targets, c_coords).to(DEVICE)

        return sinput, sc_indices, sc_targets

    def test_forward(self):
        in_channels = 24
        out_channels = 8
        sinput, sc_indices, sc_target = self.make_sample_data(in_channels, out_channels)
        model = HRNetStyleCls3S2BND128(in_channels, out_channels, config, 3).to(DEVICE)

        soutput = model(sinput, sc_indices)
        print(soutput.F)

    def test_train(self):
        in_channels = 24
        out_channels = 8
        model = HRNetStyleCls3S2BND128(in_channels, out_channels, config, 3).to(DEVICE)

        criterion = get_with_component_criterion(config)
        optimizer = initialize_optimizer(model.parameters(), config)

        torch.autograd.set_detect_anomaly(True)
        for i in range(4):

            sinput, sc_indices, sc_target = self.make_sample_data(in_channels, out_channels)
            soutput = model(sinput, sc_indices)

            optimizer.zero_grad()
            batch_loss = 0
            loss = criterion(soutput.F, torch.squeeze(sc_target.F.long()))
            print(loss)
            batch_loss += loss.item()
            loss.backward()
            torch.cuda.empty_cache()

    def test_load_ps_weights_hrnet(self):
        # ps_checkpoint = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply_100K/BuildnetVoxelization0_01Dataset/PS-HRNet3S2BD256/b32-i100000/checkpoint_HRNet3S2BD256best_acc.pth"
        # style_model = HRNetStyleCls3S2BD256(3, 10, config)
        # before = style_model.stages[1][0][0].conv1.kernel.data.cpu().numpy()
        # if os.path.exists(ps_checkpoint):
        #     logging.info('===> Loading ps weights: ' + ps_checkpoint)
        #     state = torch.load(ps_checkpoint)['state_dict']
        #     for name, param in style_model.named_parameters():
        #         if name in state:
        #             param.data = state[name]
        #     after = style_model.stages[1][0][0].conv1.kernel.data.cpu().numpy()
        #     print(before.flatten()[0:3], after.flatten()[0:3])
        ps_checkpoint = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply_100K/BuildnetVoxelization0_01Dataset/PS-HRNet3S2BD256/b32-i100000/checkpoint_HRNet3S2BD256best_acc.pth"
        style_model = HRNetStyleCls3S2BD256(3, 10, config)
        before = style_model.stages[1][0][0].conv1.kernel.data.cpu().numpy()
        style_model.load_from_ps(ps_checkpoint)
        after = style_model.stages[1][0][0].conv1.kernel.data.cpu().numpy()
        print(before.flatten()[0:3], after.flatten()[0:3])

    def test_load_ps_weights_resunet(self):
        ps_checkpoint = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/checkpoint_Res16UNet34Abest_acc.pth"
        style_model = StyleRes16UNet34A(3, 10, config)
        before = style_model.final.kernel.data.cpu().numpy()
        style_model.load_from_ps(ps_checkpoint)
        after = style_model.final.kernel.data.cpu().numpy()
        print(before.flatten()[0:3], after.flatten()[0:3])

        style_model = StyleRes16UNet34A(3, 10, config)
        before = style_model.conv1p1s2.kernel.data.cpu().numpy()
        style_model.load_from_ps(ps_checkpoint)
        after = style_model.conv1p1s2.kernel.data.cpu().numpy()
        print(before.flatten()[0:3], after.flatten()[0:3])
