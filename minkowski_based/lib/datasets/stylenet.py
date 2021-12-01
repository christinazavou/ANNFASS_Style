import logging
import os
import sys
import json
from pathlib import Path

import numpy as np
from scipy import spatial

from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, fast_hist, per_class_iu

NUM_LABELS = 9
CLASS_LABELS = ('Unknown', 'Colonial', 'Neo_classicism',
                'Modernist', 'Ottoman', 'Gothic',
                'Byzantine', 'Venetian', 'Empty')
VALID_CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
IGNORE_LABELS = (8,)

STYLENET_COLOR_MAP = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (0, 0, 0),
    6: (0, 0, 0),
    7: (0, 0, 0),
    8: (0, 0, 0)
}

NORMALIZED_FREQ = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
}

FREQ = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
}


class StylenetVoxelizationDataset(VoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.05

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi, np.pi), (-np.pi / 64, np.pi / 64))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.05, 0.05), (0, 0), (-0.05, 0.05))
  ELASTIC_DISTORT_PARAMS = ((0.1, 0.2),)
  SHIFT_PARAMS = (0.01, 0.05) # ((sigma, clip)
  N_ROTATIONS = 12

  ROTATION_AXIS = 'y'
  LOCFEAT_IDX = 2
  NUM_LABELS = NUM_LABELS
  IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
  IS_FULL_POINTCLOUD_EVAL = True
  NORM_FREQ = NORMALIZED_FREQ
  FREQ = FREQ
  class_labels = CLASS_LABELS

  # If trainval.txt does not exist, copy train.txt and add contents from val.txt
  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train.txt',
      DatasetPhase.Val: 'val.txt',
      DatasetPhase.Test: 'test.txt'
  }

  def __init__(self,
               config,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               rot_aug=False,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    data_root = config.stylenet_path
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    self.input_feat = config.input_feat.lower()
    if self.input_feat == 'normals' :
      self.NUM_IN_CHANNEL = 3
    else:
        self.NUM_IN_CHANNEL = config.num_in_channels

    super().__init__(
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        rot_aug=rot_aug,
        elastic_distortion=elastic_distortion,
        config=config)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

  def test_pointcloud_extended(self, pred_dir, use_idw=False):
    print('Running STYLENETAE full pointcloud evaluation.')
    eval_path = pred_dir
    os.makedirs(eval_path, exist_ok=True)
    # Test independently for each building
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    save_iter = 0
    for i, data_path in enumerate(self.data_paths):
      model_name = data_path.split('/')[-1][:-4]
      print("Get per point features for building {name:s} ({iter:d}/{total:d})"
            .format(name=model_name, iter=i+1, total=len(self.data_paths)))
      pred = np.load(os.path.join(pred_dir, 'pred_%04d.npy' % (i)))
      os.remove(os.path.join(pred_dir, 'pred_%04d.npy' % (i)))

      if (i+1) < save_iter:
        # save voxelized pointcloud predictions
        save_point_cloud(
            np.hstack((pred[:, :3], np.array([STYLENET_COLOR_MAP[0] for i in pred]))),
            f'{eval_path}/{model_name}_voxel.ply',
            verbose=False)

      fullply_f = self.data_root / data_path
      query_pointcloud = read_plyfile(fullply_f)
      query_xyz = query_pointcloud[:, :3]
      query_label = query_pointcloud[:, -1]
      # Run test for each room.
      pred_tree = spatial.cKDTree(pred[:, :3], leafsize=500)
      if use_idw:
        # Inverse distance weighting using 4-nn
        k, pow = 4, 2
        dist, k_nn = pred_tree.query(query_xyz, k=k)
        dist_pow = dist ** pow
        norm = np.sum(1 / dist_pow, axis=1, keepdims=True)
        norm = np.tile(norm, [1, k])
        weights = (1 / dist_pow) / norm
        assert (np.isclose(np.sum(weights, axis=1, keepdims=True), np.ones_like(norm)).all())
        feats = np.multiply(weights[..., np.newaxis], pred[k_nn, 3:-1])
        inter_feat = np.sum(feats, axis=1).astype(np.float32)
        ptc_pred = np.argmax(inter_feat, axis=1).astype(int)
        if self.IGNORE_LABELS:
          decode_label_map = {}
          for k, v in self.label_map.items():
            decode_label_map[v] = k
          ptc_pred = np.array([decode_label_map[x] for x in ptc_pred], dtype=np.int)
      else:
        _, result = pred_tree.query(query_xyz)
        ptc_pred = pred[result, 3].astype(int)

      # Save per point output features to be used for evaluation
      ### WARNING: IDW has to be TRUE
      np.save(os.path.join(eval_path, model_name + ".npy"), inter_feat)

class StylenetVoxelization0_02Dataset(StylenetVoxelizationDataset):
  VOXEL_SIZE = 0.02


class StylenetVoxelization0_01Dataset(StylenetVoxelizationDataset):
  VOXEL_SIZE = 0.01


class StylenetVoxelization0_005Dataset(StylenetVoxelizationDataset):
  VOXEL_SIZE = 0.005
