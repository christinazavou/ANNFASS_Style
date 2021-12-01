from unittest import TestCase

import sys

from config import get_config
from lib.export_features_extended import get_feats, write_ply_with_features
from lib.datasets import load_dataset

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

testargs = ["python test_resunet_ae.py",
            "--stylenet_path",
            "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_splits_march/ply_100K_cnscr/fold0/split_test",
            "--dataset", "StylenetComponentVoxelization0_01Dataset",
            "--input_feat", "coords",
            "--export_feat", "True",
            "--model", "HRNetAE1S2BD128",
            "--weights",
            "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_minkowski_ae/annfass_ply_100K_cnscr/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b5-i500/checkpoint_HRNetAE1S2BD128best_loss.pth",
            "--chamfer_loss", "True",
            "--return_transformation", "True",
            "--save_pred_dir",
            "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_minkowski_ae/annfass_ply_100K_cnscr/fold0/StylenetXYZAEVoxelization0_01Dataset/AE-HRNetAE1S2BD128/b5-i500/export_feat_ply"]
with patch.object(sys, 'argv', testargs):
    config = get_config()


class Test(TestCase):
    def test_get_feats(self):
        DatasetClass = load_dataset(config.dataset)
        get_feats(DatasetClass, config)

    def test_write_ply_with_normals_and_others(self):
        import numpy as np
        vertices = np.random.random((10, 3))
        features = np.random.random((10, 6))
        labels = np.random.randint(0, 2, (10))
        write_ply_with_features("outputs/test.ply", vertices, features, labels)
        write_ply_with_features("outputs/test1.ply", vertices, features)
