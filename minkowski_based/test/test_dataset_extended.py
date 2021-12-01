import os
from unittest import TestCase
from MinkowskiEngine.SparseTensor import SparseTensor
import torch
import numpy as np

from lib.dataset_extended import VoxelizationDatasetBase, initialize_data_loader
from lib.datasets import load_dataset

from config import get_config

import sys

from lib.transforms_extended import get_component_indices_matrix
from lib.utils import get_class_counts, get_class_weights, save_class_weights
from models.modules.non_trainable_layers import get_average_per_component_t

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


class TestVoxelizationDatasetBase(TestCase):

    def test_load_ply(self):
        root_path = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/samplePoints"
        data_paths = [root_path + "/ply_100K_cnscr/01_Cathedral_of_Holy_Wisdom.ply",
                      root_path + "/ply_100K_cnscr/02_Panagia_Chrysaliniotissa.ply"]
        vdb = VoxelizationDatasetBase(data_paths)
        coords, feats, labels, c_centers, components = vdb.load_ply(1, 'normals')
        print(np.unique(components))
        print(np.unique(labels))
        coords, feats, labels, c_centers, components = vdb.load_ply(1, 'normals', 'rnv')
        print(np.unique(components))
        print(np.unique(labels))
        coords, feats, labels, c_centers, components = vdb.load_ply(1, 'normals', 'rov')
        print(np.unique(components))
        print(np.unique(labels))

    def test_inputdata(self):

        testargs = ["python test_dataset_extended.py",
                    "--stylenet_path",
                    "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_splits_march/ply_100K_cnscr/fold0/split_train_val_test",
                    "--dataset", "StylenetComponentVoxelization0_01Dataset",
                    "--input_feat", "normals",
                    "--batch_size", "4"]
        with patch.object(sys, 'argv', testargs):
            config = get_config()

        DatasetClass = load_dataset(config.dataset)

        custom_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            phase=config.test_phase,
            num_workers=config.num_workers,
            augment_data=False,
            shift=config.shift,
            jitter=config.jitter,
            rot_aug=config.rot_aug,
            scale=config.scale,
            shuffle=False,
            repeat=False,
            batch_size=config.batch_size,
            limit_numpoints=config.train_limit_numpoints)

        data_iter = custom_data_loader.__iter__()
        for i in range(3):
            coords, input, target, component_ids, component_names = data_iter.next()
            sinput = SparseTensor(input, coords).to('cuda')
            print(sinput)

            cimat, cnames = get_component_indices_matrix(component_ids, component_names)

            c_i_mat_t = torch.tensor(cimat)
            c_coords_t = get_average_per_component_t(c_i_mat_t, coords)
            c_input_t = get_average_per_component_t(c_i_mat_t, input)
            c_target_t = get_average_per_component_t(c_i_mat_t, torch.unsqueeze(target, 1), is_target=True)
            print(c_coords_t)
            print(c_input_t)
            print(c_target_t)

    def test_class_weights_balanced(self):

        def run_one_split(split_id):
            testargs = ["python test_dataset_extended.py",
                        "--stylenet_path",
                        f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/combined_splits_final_unique/ANNFASS_Buildings_may_ply_cut100.0KCombined_Buildings_ply_cut100.0K_wg/{split_id}/split_train_val_test",
                        "--dataset", "StylenetComponentVoxelization0_01Dataset",
                        "--input_feat", "coords",
                        "--ignore_label", "-1",
                        "--class_balanced_loss", "True",
                        "--class_balanced_beta", "0.999"]
            with patch.object(sys, 'argv', testargs):
                config = get_config()

            DatasetClass = load_dataset(config.dataset)

            dataset = DatasetClass(
                config,
                cache=config.cache_data,
                # phase=config.train_phase)
                phase=config.val_phase)

            print(f"Split {split_id}")
            weights = get_class_weights(config, dataset)
            return weights

        out_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/combined_splits_final_unique/ANNFASS_Buildings_may_ply_cut100.0KCombined_Buildings_ply_cut100.0K_wg.weights.balanced"
        os.makedirs(out_dir, exist_ok=True)
        for directory in os.listdir("/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/combined_splits_final_unique/ANNFASS_Buildings_may_ply_cut100.0KCombined_Buildings_ply_cut100.0K_wg"):
            out_file = os.path.join(out_dir, f"{directory}.npy")
            class_weights = run_one_split(directory, )
            save_class_weights(class_weights, out_file)

    def test_class_weights_inv_freq(self):

        def run_one_split(split_id):
            testargs = ["python test_dataset_extended.py",
                        "--stylenet_path",
                        f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/combined_splits_final_unique/ANNFASS_Buildings_may_ply_cut100.0KCombined_Buildings_ply_cut100.0K_wg/{split_id}/split_train_val_test",
                        "--dataset", "StylenetComponentVoxelization0_01Dataset",
                        "--input_feat", "coords",
                        "--ignore_label", "-1",
                        "--inv_freq_class_weight", "True"]
            with patch.object(sys, 'argv', testargs):
                config = get_config()

            DatasetClass = load_dataset(config.dataset)

            dataset = DatasetClass(
                config,
                cache=config.cache_data,
                # phase=config.train_phase)
                phase=config.val_phase)

            print(f"Split {split_id}")
            weights = get_class_weights(config, dataset)
            return weights

        out_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/combined_splits_final_unique/ANNFASS_Buildings_may_ply_cut100.0KCombined_Buildings_ply_cut100.0K_wg.weights.inv_freq"
        os.makedirs(out_dir, exist_ok=True)
        for directory in os.listdir("/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/combined_splits_final_unique/ANNFASS_Buildings_may_ply_cut100.0KCombined_Buildings_ply_cut100.0K_wg"):
            out_file = os.path.join(out_dir, f"{directory}.npy")
            class_weights = run_one_split(directory, )
            save_class_weights(class_weights, out_file)
