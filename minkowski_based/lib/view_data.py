import os
import argparse
import numpy as np
import open3d as o3d

from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from config import str2bool


# BuildNet classes
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                   29, 30, 31, 32, 33)
COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 69, 0),
    2: (0, 0, 255),
    3: (57, 96, 115),
    4: (75, 0, 140),
    5: (250, 128, 114),
    6: (127, 0, 0),
    7: (214, 242, 182),
    8: (13, 33, 51),
    9: (32, 64, 53),
    10: (255, 64, 64),
    11: (96, 185, 191),
    12: (61, 64, 16),
    13: (115, 61, 0),
    14: (64, 0, 0),
    15: (153, 150, 115),
    16: (255, 0, 255),
    17: (57, 65, 115),
    18: (85, 61, 242),
    19: (191, 48, 105),
    20: (48, 16, 64),
    21: (255, 145, 128),
    22: (153, 115, 145),
    23: (255, 191, 217),
    24: (0, 170, 255),
    25: (138, 77, 153),
    26: (64, 255, 115),
    27: (140, 110, 105),
    28: (204, 0, 255),
    29: (178, 71, 0),
    30: (255, 187, 221),
    31: (13, 211, 255),
    32: (0, 64, 26),
    33: (195, 230, 57),
    255: (0., 0., 0.)
}



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='StylenetVoxelization0_01Dataset')
  parser.add_argument('--stylenet_path', type=str, default='/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_reconstruction_splits/ply_100K/split_train_val_test_debug')
  parser.add_argument('--split', type=str, default='train')
  parser.add_argument('--input_feat', type=str, default='normals')
  parser.add_argument('--normalize_coords', type=str2bool, default=False, help='Normalize coordinates')
  parser.add_argument('--normalize_method', type=str, default="sphere", help='Methot do normalize coordinates')
  parser.add_argument('--ignore_label', type=int, default=255)
  parser.add_argument('--return_transformation', type=str2bool, default=False)
  parser.add_argument('--prefetch_data', type=str2bool, default=False)
  parser.add_argument('--cache_data', type=str2bool, default=False)
  parser.add_argument('--data_aug_color_trans_ratio', type=float, default=0.10, help='Color translation range')
  parser.add_argument('--data_aug_color_jitter_std', type=float, default=0.05, help='STD of color jitter')
  # parser.add_argument('--normalize_color', type=str2bool, default=True)
  config = parser.parse_args()

  # Load dataset class
  DatasetClass = load_dataset(config.dataset)

  # Get data split
  data_loader = initialize_data_loader(
      DatasetClass,
      config,
      num_workers=1,
      phase=config.split,
      augment_data=False,
      shift=False,
      jitter=False,
      rot_aug=False,
      scale=False,
      shuffle=False,
      repeat=False,
      batch_size=1,
      limit_numpoints=1000000)

  # View data
  data_loader.dataset.IGNORE_LABELS = None
  data_iter = data_loader.__iter__()
  for sub_iter in range(len(data_loader)):
    voxel_coords, voxel_input, voxel_target = data_iter.next()

    # Create a point cloud file for the voxelized version
    voxel_coords = voxel_coords[:, 1:].cpu().numpy()
    assert(not np.isnan(voxel_coords).any())
    voxel_target = voxel_target.cpu().numpy()
    # Map color
    # voxel_colors = np.array([COLOR_MAP[l] for l in voxel_target])
    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(voxel_coords * data_loader.dataset.VOXEL_SIZE)
    # voxel_pcd.colors = o3d.utility.Vector3dVector(voxel_colors / 255)

    # Move the original point cloud
    coords, input, target, _ = data_loader.dataset.load_ply(sub_iter, input_feat='normals')
    # colors = np.array([COLOR_MAP[l] for l in target])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords  + np.array([2.5, 0, 0]))
    # pcd.colors = o3d.utility.Vector3dVector(colors / 255)

    # Visualize the voxelized point cloud
    o3d.visualization.draw_geometries([voxel_pcd, pcd])

