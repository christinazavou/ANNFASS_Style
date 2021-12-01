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

CLASS_LABELS = ('wall', 'window', 'vehicle', 'roof', 'plant_tree', 'door', 'tower_steeple', 'furniture',
                'ground_grass', 'beam_frame', 'stairs', 'column', 'railing_baluster', 'floor', 'chimney', 'ceiling',
                'fence', 'pond_pool', 'corridor_path', 'balcony_patio', 'garage', 'dome', 'road', 'entrance_gate',
                'parapet_merlon', 'buttress', 'dormer', 'lantern_lamp', 'arch', 'awning', 'shutters') #, 'ramp', 'canopy_gazebo')

VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                   29, 30, 31)#, 32, 33)
BUILDNET_COLOR_MAP = {
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
    33: (195, 230, 57)
}

NORMALIZED_FREQ = {
    1: 1,
    2: 9.3881,
    3: 73.6199,
    4: 2.09293,
    5: 20.0008,
    6: 52.77571,
    7: 15.26988,
    8: 103.09889,
    9: 3.43945,
    10: 35.35736,
    11: 78.68262,
    12: 64.82034,
    13: 129.65081,
    14: 7.11018,
    15: 108.05064,
    16: 17.01521,
    17: 29.84063,
    18: 30.18609,
    19: 29.79754,
    20: 51.42419,
    21: 155.69354,
    22: 49.74032,
    23: 22.5733,
    24: 798.24279,
    25: 145.46404,
    26: 146.36735,
    27: 363.90137,
    28: 5364.96081,
    29: 288.8162,
    30: 670.37904,
    31: 1363.54926,
    # 32: 1460.59285,
    # 33: 2155.44005
}

FREQ = {
    0: 42495545,
    1: 48560569,
    2: 4245818,
    3: 670675,
    4: 19884263,
    5: 2301037,
    6: 988026,
    7: 3181645,
    8: 449339,
    9: 13343544,
    10: 1704418,
    11: 784710,
    12: 963960,
    13: 427006,
    14: 6533642,
    15: 586404,
    16: 2614624,
    17: 1671562,
    18: 1439760,
    19: 1559239,
    20: 897539,
    21: 302712,
    22: 973209,
    23: 2154294,
    24: 58982,
    25: 341471,
    26: 372939,
    27: 140856,
    28: 29077,
    29: 207131,
    30: 88028,
    31: 27976
}


class BuildnetVoxelizationDataset(VoxelizationDataset):

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
  NUM_LABELS = 34  # Will be converted to 31 as defined in IGNORE_LABELS.
  IGNORE_LABELS = (0, 32, 33)
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
    data_root = config.buildnet_path
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    self.input_feat = config.input_feat.lower()
    if self.input_feat == 'rgb' or self.input_feat == 'normals' :
      self.NUM_IN_CHANNEL = 3
    elif self.input_feat == 'rgba':
      self.NUM_IN_CHANNEL = 4
    elif self.input_feat == 'normals_rgb':
      self.NUM_IN_CHANNEL = 6
    elif self.input_feat == 'normals_rgba':
      self.NUM_IN_CHANNEL = 7
    else:
      print("Unknown input features {feat:s}" .format(feat=self.input_feat))
      exit(-1)

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

  def _augment_locfeat(self, pointcloud):
    # Assuming that pointcloud is xyzrgb(...), append location feat.
    pointcloud = np.hstack(
        (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
         pointcloud[:, 6:]))
    return pointcloud

  def test_pointcloud(self, pred_dir):
    print('Running full pointcloud evaluation.')
    eval_path = os.path.join(pred_dir, 'fulleval')
    os.makedirs(eval_path, exist_ok=True)
    # Join room by their area and room id.
    # Test independently for each room.
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
    for i, data_path in enumerate(self.data_paths):
      room_id = self.get_output_id(i)
      pred = np.load(os.path.join(pred_dir, 'pred_%04d.npy' % (i)))

      # save voxelized pointcloud predictions
      save_point_cloud(
          np.hstack((pred[:, :3], np.array([BUILDNET_COLOR_MAP[i] for i in pred[:, -1]]))),
          f'{eval_path}/{room_id}_voxel.ply',
          binary=False,
          verbose=False)

      fullply_f = self.data_root / data_path
      query_pointcloud = read_plyfile(fullply_f)
      query_xyz = query_pointcloud[:, :3]
      query_label = query_pointcloud[:, -1]
      # Run test for each room.
      pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
      _, result = pred_tree.query(query_xyz)
      ptc_pred = pred[result, 3].astype(int)
      # Save prediciton in txt format for submission.
      np.savetxt(f'{eval_path}/{room_id}.txt', ptc_pred, fmt='%i')
      # Save prediciton in colored pointcloud for visualization.
      save_point_cloud(
          np.hstack((query_xyz, np.array([BUILDNET_COLOR_MAP[i] for i in ptc_pred]))),
          f'{eval_path}/{room_id}.ply',
          binary=False,
          verbose=False)
      # Evaluate IoU.
      if self.IGNORE_LABELS is not None:
        ptc_pred = np.array([self.label_map[x] for x in ptc_pred], dtype=np.int)
        query_label = np.array([self.label_map[x] for x in query_label], dtype=np.int)
      hist += fast_hist(ptc_pred, query_label, self.NUM_LABELS)
    ious = per_class_iu(hist) * 100
    print('mIoU: ' + str(np.nanmean(ious)) + '\n'
          'Class names: ' + ', '.join(CLASS_LABELS) + '\n'
          'IoU: ' + ', '.join(np.round(ious, 2).astype(str)))

  def test_pointcloud_extended(self, pred_dir, use_idw=False):
    print('Running BUILDNET full pointcloud evaluation.')
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
            np.hstack((pred[:, :3], np.array([BUILDNET_COLOR_MAP[i] for i in pred[:, -1]]))),
            f'{eval_path}/{model_name}_voxel.ply',
            binary=False,
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

      if (i + 1) < save_iter:
        # Save prediciton in colored pointcloud for visualization.
        save_point_cloud(
            np.hstack((query_xyz, np.array([BUILDNET_COLOR_MAP[i] for i in query_label]))),
            f'{eval_path}/{model_name}_gt.ply',
            binary=False,
            verbose=False)
        save_point_cloud(
            np.hstack((query_xyz, np.array([BUILDNET_COLOR_MAP[i] for i in ptc_pred]))),
            f'{eval_path}/{model_name}_pred.ply',
            binary=False,
            verbose=False)

      # Save per point output features to be used for evaluation
      ### WARNING: IDW has to be TRUE
      np.save(os.path.join(eval_path, model_name + ".npy"), inter_feat)


class BuildnetVoxelization0_02Dataset(BuildnetVoxelizationDataset):
  VOXEL_SIZE = 0.02

class BuildnetVoxelization0_01Dataset(BuildnetVoxelizationDataset):
  VOXEL_SIZE = 0.01

class BuildnetVoxelization0_005Dataset(BuildnetVoxelizationDataset):
  VOXEL_SIZE = 0.005
