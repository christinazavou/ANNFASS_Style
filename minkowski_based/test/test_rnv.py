from unittest import TestCase
from MinkowskiEngine import SparseTensor
from torch import nn

from lib.dataset_extended import initialize_data_loader
from lib.datasets import load_dataset

from config import get_config

import sys

from models import load_model

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


class MyTestCase(TestCase):

    def test_inputdata(self):

        testargs = ["python test_rnv.py",
                    "--stylenet_path",
                    "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_splits_march/ply_100K_cnscr/fold0/split_train_test",
                    "--dataset", "StylenetRNVVoxelization0_01Dataset",
                    "--input_feat", "normals",
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
        coords, input, target, _, _ = data_iter.next()
        print(coords, coords.shape)
        print(input, input.shape)
        print(target, target.shape)
        sinput = SparseTensor(input, coords).to('cpu')
        target = target.long().to('cpu')
        NetClass = load_model(config.model)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        model = NetClass(3, 2, config).to('cpu')
        out = model(sinput)
        loss = criterion(out.F, target)
        print(loss)
