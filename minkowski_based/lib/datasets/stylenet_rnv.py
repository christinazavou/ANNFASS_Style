import logging
import os
from pathlib import Path

import numpy as np

from lib.dataset import DatasetPhase, str2datasetphase_type
from lib.dataset_extended import VoxelizationDataset
from lib.utils import read_txt


class StylenetRidgeValleyVoxelizationDataset(VoxelizationDataset):

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
  IS_FULL_POINTCLOUD_EVAL = True
  NORM_FREQ = None
  FREQ = None

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
    if self.input_feat == 'normals':
      if self.AUGMENT_COORDS_TO_FEATS:
          self.NUM_IN_CHANNEL = 6
      else:
          self.NUM_IN_CHANNEL = 3
    else:
      assert self.input_feat == 'coords' and self.AUGMENT_COORDS_TO_FEATS
      self.NUM_IN_CHANNEL = 3
    self.label_feat = self.LABEL_FEAT

    super().__init__(
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=-1,  # note this (since components can be as many as arbitrary positive number
        # and we need to voxelize component ids)!
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        rot_aug=rot_aug,
        elastic_distortion=elastic_distortion,
        config=config)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

  def test_pointcloud(self, pred_dir):
    pass


  def get_component_indices_matrix(self, component_idx):
    component_idx_list = sorted(list(set(component_idx) - {-1}))
    num_components = len(component_idx_list)
    num_points = len(component_idx)
    component_indices = np.zeros((num_components, num_points))
    for point_idx, c_idx in enumerate(component_idx):
      if c_idx != -1:
        c_idx = component_idx_list.index(c_idx)
        component_indices[c_idx, point_idx] = 1
    assert not any(np.sum(component_indices, 1) == 0)
    return component_indices

  def get_component_average(self, component_indices, point_vectors):
    outfeat = np.matmul(np.expand_dims(component_indices, -2), np.expand_dims(point_vectors, -3))
    outfeat = np.squeeze(outfeat, 1) / np.expand_dims(np.sum(component_indices, 1), -1)
    return outfeat


class Stylenet_RNV_Voxelization0_01Dataset(StylenetRidgeValleyVoxelizationDataset):
  VOXEL_SIZE = 0.01
  IGNORE_LABELS = None
  LABEL_FEAT = 'rnv'
  NUM_LABELS = 2
  CLASS_LABELS = ('NotRidgeValley', 'RidgeOrValley')
  VALID_CLASS_IDS = (0, 1)
  class_labels = CLASS_LABELS


class Stylenet_ROV_Voxelization0_01Dataset(StylenetRidgeValleyVoxelizationDataset):
  VOXEL_SIZE = 0.01
  IGNORE_LABELS = None
  LABEL_FEAT = 'rov'
  NUM_LABELS = 3
  CLASS_LABELS = ('NotRidgeValley', 'Ridge', 'Valley')
  VALID_CLASS_IDS = (0, 1, 2)
  class_labels = CLASS_LABELS
