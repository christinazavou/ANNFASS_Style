import logging
import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from scipy import spatial

from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, calculate_iou, calculate_shape_iou, calculate_part_iou
import lib.transforms as t


NUM_SEG = {'Bed': 15,
           'Bottle': 9,
           'Chair': 39,
           'Clock': 11,
           'Dishwasher': 7,
           'Display': 4,
           'Door': 5,
           'Earphone': 10,
           'Faucet': 12,
           'Knife': 10,
           'Lamp': 41,
           'Microwave': 6,
           'Refrigerator': 7,
           'StorageFurniture': 24,
           'Table': 51,
           'TrashCan': 11,
           'Vase': 6}


class PartnetVoxelizationDataset(VoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.05

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = (-5 * np.pi / 180.0, 5 * np.pi / 180) #, (-np.pi, np.pi), (-np.pi / 64, np.pi / 64))
  JITTER_AUGMENTATION_BOUND = (0.25, 0.25, 0.25)
  SCALE_AUGMENTATION_BOUND = (0.75, 1.25)
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.05, 0.05), (0, 0), (-0.05, 0.05))
  ELASTIC_DISTORT_PARAMS = ((0.1, 0.2),)
  SHIFT_PARAMS = (0.01, 0.05) # ((sigma, clip)
  N_ROTATIONS = 1

  ROTATION_AXIS = 'y'
  LOCFEAT_IDX = 2
  IS_FULL_POINTCLOUD_EVAL = True

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

    # Init labels and color map
    self.NUM_LABELS = NUM_SEG[config.partnet_category]
    self.VALID_CLASS_IDS = tuple(range(self.NUM_LABELS))
    self.IGNORE_LABELS = tuple(set(range(self.NUM_LABELS)) - set(self.VALID_CLASS_IDS)) # in case we want to remove the zero label
    cmap = plt.cm.get_cmap("hsv", self.NUM_LABELS)
    self.PARTNET_COLOR_MAP = dict(zip(range(self.NUM_LABELS),
                                      [tuple([int(cmap(i)[0]*255), int(cmap(i)[1]*255), int(cmap(i)[2]*255)])
                                       for i in range(self.NUM_LABELS)]))

    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    data_root = os.path.join(config.partnet_path, config.partnet_category)
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    self.input_feat = config.input_feat.lower()
    if self.input_feat == 'normals' :
      self.NUM_IN_CHANNEL = 3
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
    print('Running PartNet full pointcloud evaluation.')
    eval_path = pred_dir
    os.makedirs(eval_path, exist_ok=True)
    # Test independently for each building
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    top_k = 5
    best_iou_model = np.zeros((top_k,))
    best_iou_model[:] = 0.000000001
    best_model_voxels_pred, best_model_points_gt, best_model_points_pred, best_model_fn = [[] for _ in range(top_k)], \
                                                                                          [[] for _ in range(top_k)], \
                                                                                          [[] for _ in range(top_k)], \
                                                                                          [[] for _ in range(top_k)]

    ious = {}

    for i, data_path in enumerate(self.data_paths):
      model_name = data_path.split('/')[-1][:-4]
      print("Evaluate {name:s} ({iter:d}/{total:d})"
            .format(name=model_name, iter=i+1, total=len(self.data_paths)))
      pred = np.load(os.path.join(pred_dir, 'pred_%04d.npy' % (i)))
      os.remove(os.path.join(pred_dir, 'pred_%04d.npy' % (i)))

      fullply_f = self.data_root / data_path
      pointcloud = read_plyfile(fullply_f)
      points = pointcloud[:, :3]
      if self.normalize:
        points = t.normalize_coords(points, self.normalize_method)
      point_gt_labels = pointcloud[:, -1]
      pred_tree = spatial.cKDTree(pred[:, :3], leafsize=500)
      # Inverse distance weighting using 4-nn
      k, pow = 4, 2
      dist, k_nn = pred_tree.query(points, k=k)
      dist_pow = dist ** pow
      norm = np.sum(1 / dist_pow, axis=1, keepdims=True)
      norm = np.tile(norm, [1, k])
      weights = (1 / dist_pow) / norm
      assert (np.isclose(np.sum(weights, axis=1, keepdims=True), np.ones_like(norm)).all())
      feats = np.multiply(weights[..., np.newaxis], pred[k_nn, 3:-1])
      inter_feat = np.sum(feats, axis=1).astype(np.float32)
      point_pred_labels = np.argmax(inter_feat, axis=1).astype(int)
      if self.IGNORE_LABELS:
        decode_label_map = {}
        for k, v in self.label_map.items():
          decode_label_map[v] = k
        point_pred_labels = np.array([decode_label_map[x] for x in point_pred_labels], dtype=np.int)
      point_gt_labels = point_gt_labels[:, np.newaxis]
      point_pred_labels = point_pred_labels[:, np.newaxis]

      # Calculate iou
      if not np.array_equal(np.unique(point_gt_labels), np.array([0])):
        ious[model_name] = calculate_iou(ground=point_gt_labels, prediction=point_pred_labels, num_labels=self.NUM_LABELS)

        # Save best and worst model
        label_iou = ious[model_name]["label_iou"]
        s_iou = np.nan_to_num(np.sum([v for v in label_iou.values()]) / float(len(label_iou)))
        if s_iou > best_iou_model[-1]:
          best_iou_model[top_k - 1] = s_iou
          best_model_points_gt[top_k - 1] = np.hstack((points, point_gt_labels))
          best_model_points_pred[top_k - 1] = np.hstack((points, point_pred_labels))
          best_model_voxels_pred[top_k - 1] = np.hstack((pred[:, :3], pred[:, -1][:, np.newaxis]))
          best_model_fn[top_k - 1] = model_name
          sort_idx = np.argsort(1 / np.asarray(best_iou_model)).tolist()
          best_iou_model = best_iou_model[sort_idx]
          best_model_points_gt = [best_model_points_gt[idx] for idx in sort_idx]
          best_model_points_pred = [best_model_points_pred[idx] for idx in sort_idx]
          best_model_voxels_pred = [best_model_voxels_pred[idx] for idx in sort_idx]
          best_model_fn = [best_model_fn[idx] for idx in sort_idx]

    # Calculate avg point part and shape IoU
    shape_iou = calculate_shape_iou(ious=ious) * 100
    part_iou = calculate_part_iou(ious=ious, num_labels=self.NUM_LABELS) * 100

    # Save best
    buf = ''
    for i in range(top_k):
      buf += "Best model iou: " + str(best_iou_model[i]) + ", " + best_model_fn[i] + '\n'
      # Save colored pointcloud for visualization.
      save_point_cloud(
          np.hstack((best_model_points_gt[i][:, :3], np.array([self.PARTNET_COLOR_MAP[i] for i in best_model_points_gt[i][:, -1]]))),
          f'{eval_path}/{best_model_fn[i]}_gt.ply',
          verbose=False)
      save_point_cloud(
        np.hstack((best_model_points_pred[i][:, :3],
                  np.array([self.PARTNET_COLOR_MAP[i] for i in best_model_points_pred[i][:, -1]]))),
        f'{eval_path}/{best_model_fn[i]}_pred.ply',
        verbose=False)
      save_point_cloud(
        np.hstack((best_model_voxels_pred[i][:, :3],
                  np.array([self.PARTNET_COLOR_MAP[i] for i in best_model_voxels_pred[i][:, -1]]))),
        f'{eval_path}/{best_model_fn[i]}_voxels_pred.ply',
        verbose=False)

    # Log results
    buf += "Shape IoU: " + str(np.round(shape_iou, 2)) + '\n' \
                                                        "Part IoU: " + str(np.round(part_iou, 2))

    print(buf)
    with open(os.path.join(eval_path, "results_log.txt"), 'w') as fout_txt:
      fout_txt.write(buf)


class PartnetVoxelization0_02Dataset(PartnetVoxelizationDataset):
  VOXEL_SIZE = 0.02

class PartnetVoxelization0_01Dataset(PartnetVoxelizationDataset):
  VOXEL_SIZE = 0.01

class PartnetVoxelization0_005Dataset(PartnetVoxelizationDataset):
  VOXEL_SIZE = 0.005
