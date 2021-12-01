from unittest import TestCase
from MinkowskiEngine.SparseTensor import SparseTensor, _get_coords_key
import torch
import numpy as np

from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from models.modules.common import conv

from config import get_config

import sys
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


class Test(TestCase):

    def test_gather(self):
        features = np.array([
            [1,  2,  3,  4,  5,  6],
            [7,  8,  9,  10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30]]).astype(float)
        features = torch.tensor(features).cuda()
        components = np.array([1, 1, 2, 2, 3])
        components = torch.tensor(components).cuda()
        indices = components.unsqueeze(-1)
        res = torch.gather(features, 0, indices)
        print(res)
        components = [[1,2], [3,4], [0]]
        for component_indices in components:
            component_features = torch.mean(features[component_indices, :], 0)
            print(component_features)

    def test_gather_with_matrix(self):
        features = np.array([
            [1,  2,  3,  4,  5,  6],
            [7,  8,  9,  10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30]]).astype(float)
        features = torch.tensor(features).cuda()
        components = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0]])
        components = torch.tensor(components).cuda()
        res = torch.matmul(torch.unsqueeze(components.float(), dim=-2), torch.unsqueeze(features, dim=-3).float())
        res = torch.div(torch.squeeze(res, dim=1), torch.unsqueeze(torch.sum(components, 1), dim=-1))
        print(res)

    def test_conv(self):
        c = conv(6, 6, 2, D=3)
        coords = np.array([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9]]).astype(int)
        coords = torch.tensor(coords).int()
        features = np.array([
            [1,  2,  3,  4,  5,  6],
            [7,  8,  9,  10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36]]).astype(float)
        features = torch.tensor(features).float()

        sinput = SparseTensor(features, coords).to('cuda')
        print(sinput.coords_key)

        # ck = _get_coords_key(features, coords)
        # print(ck)

    def test_inputdata(self):

        testargs = ["python test_non_trainable_layers.py",
                    "--stylenet_path",
                    "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_reconstruction_splits/ply_100K/split_train_val_test_debug",
                    "--dataset", "StylenetXYZAEVoxelization0_01Dataset",
                    "--input_feat", "coords",
                    "--batch_size", "4"]
        with patch.object(sys, 'argv', testargs):
            config = get_config()

        DatasetClass = load_dataset(config.dataset)

        train_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            phase=config.train_phase,
            num_workers=config.num_workers,
            augment_data=False,
            shift=config.shift,
            jitter=config.jitter,
            rot_aug=config.rot_aug,
            scale=config.scale,
            shuffle=True,
            repeat=True,
            batch_size=config.batch_size,
            limit_numpoints=config.train_limit_numpoints)

        data_iter = train_data_loader.__iter__()
        coords, input, target = data_iter.next()
        print(coords, coords.shape)
        print(input, input.shape)
        sinput = SparseTensor(input, coords).to('cuda')
        print(sinput.C)
        print(sinput.F)
