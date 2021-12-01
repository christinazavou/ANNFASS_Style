from unittest import TestCase
from MinkowskiEngine import SparseTensor
import torch

from config import get_config
from lib.utils import get_torch_device
from models.resunet_ae import SmallNetAE


import sys

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


testargs = ["python test_resunet_ae.py",
            "--stylenet_path", "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_reconstruction_splits/ply_100K/split_train_val_test_debug",
            "--dataset", "StylenetXYZAEVoxelization0_01Dataset",
            "--input_feat", "coords"]
with patch.object(sys, 'argv', testargs):
    config = get_config()


class TestResUNetSmallStyleAE(TestCase):

    def test_kati(self):
        samples = 100000
        coords = torch.randn(samples, 4).cuda()
        input = torch.randn(samples, 3).cuda()

        device = get_torch_device(True)
        sinput = SparseTensor(input, coords).to(device)

        model = SmallNetAE(3, 3, config, 3)
        model = model.to(device)

        soutput = model(sinput)
        assert sinput.F.shape == soutput.F.shape
