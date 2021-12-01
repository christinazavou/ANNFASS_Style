from unittest import TestCase
import torch
import sys
import numpy as np

from MinkowskiEngine import SparseTensor

from lib.solvers import initialize_optimizer
from lib.utils import get_with_component_criterion, count_parameters
from models.hrnet_style_cls import HRNetStyleCls3S2BD128, HRNetStyleCls3S2BD256, HRNetStyleCls1S2BD128, \
    HRNetStyleCls3S3BND256, HRNetStyleCls3S3BNDF4256, HRNetStyleCls3S2BND128
from models.hrnet_style_cls_fromps import HRNetStyleClsFromPs3S2BD256
from models.res16unet_style_cls_fromps import StyleClsFromPsRes16UNet34A
from models.resunet_style_cls import SmallNetStyleCls
from config import get_config


try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


testargs = ["python test_resunet_style_cls.py",
            "--stylenet_path", "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_reconstruction_splits/ply_10K/split_train_val_test_debug",
            "--dataset", "StylenetXYZAEVoxelization0_01Dataset",
            "--input_feat", "coords",
            "--chamfer_loss", "True"]
with patch.object(sys, 'argv', testargs):
    config = get_config()


# DEVICE = 'cuda'
DEVICE = 'cpu'


class TestSmallNetStyleCls(TestCase):

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
        in_channels = 3
        out_channels = 11
        sinput, sc_indices, sc_target = self.make_sample_data(in_channels, out_channels)

        model = SmallNetStyleCls(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("SmallNetStyleCls", soutput.F.shape, count_parameters(model))

        model = HRNetStyleCls3S2BD128(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("HRNetStyleCls3S2BD128", soutput.F.shape, count_parameters(model))

        model = HRNetStyleCls3S2BD256(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("HRNetStyleCls3S2BD256", soutput.F.shape, count_parameters(model))

        model = HRNetStyleCls1S2BD128(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("HRNetStyleCls1S2BD128", soutput.F.shape, count_parameters(model))

        model = HRNetStyleCls3S3BND256(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("HRNetStyleCls3S3BND256", soutput.F.shape, count_parameters(model))

        model = HRNetStyleCls3S3BNDF4256(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("HRNetStyleCls3S3BNDF4256", soutput.F.shape, count_parameters(model))

        model = HRNetStyleCls3S2BND128(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("HRNetStyleCls3S2BND128", soutput.F.shape, count_parameters(model))

        model = HRNetStyleClsFromPs3S2BD256(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("HRNetStyleClsFromPs3S2BD256", soutput.F.shape, count_parameters(model))

        model = StyleClsFromPsRes16UNet34A(in_channels, out_channels, config, 3).to(DEVICE)
        soutput = model(sinput, sc_indices)
        print("StyleFromPsRes16UNet34A", soutput.F.shape, count_parameters(model))


    def test_train(self):
        in_channels = 24
        out_channels = 8
        model = SmallNetStyleCls(in_channels, out_channels, config, 3).to(DEVICE)

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
