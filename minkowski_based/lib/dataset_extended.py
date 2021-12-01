from abc import ABC
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import sys
import random
import numpy as np
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

from plyfile import PlyData
import lib.transforms_extended as t
from lib.dataloader import InfSampler
from lib.voxelizer import Voxelizer
from lib.dataset import str2datasetphase_type


def cache(func):

  def wrapper(self, *args, **kwargs):
    # Assume that args[0] is index
    index = args[0]
    if self.cache:
      if index not in self.cache_dict[func.__name__]:
        results = func(self, *args, **kwargs)
        self.cache_dict[func.__name__][index] = results
      return self.cache_dict[func.__name__][index]
    else:
      return func(self, *args, **kwargs)

  return wrapper


class DictDataset(Dataset, ABC):

  IS_FULL_POINTCLOUD_EVAL = False

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/'):
    """
    data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
    """
    Dataset.__init__(self)

    # Allows easier path concatenation
    if not isinstance(data_root, Path):
      data_root = Path(data_root)

    self.data_root = data_root
    self.data_paths = data_paths

    self.prevoxel_transform = prevoxel_transform
    self.input_transform = input_transform
    self.target_transform = target_transform

    # dictionary of input
    self.data_loader_dict = {
        'input': (self.load_input, self.input_transform),
        'target': (self.load_target, self.target_transform)
    }

    # For large dataset, do not cache
    self.cache = cache
    self.cache_dict = defaultdict(dict)
    self.loading_key_order = ['input', 'target']

  def load_input(self, index):
    raise NotImplementedError

  def load_target(self, index):
    raise NotImplementedError

  def get_classnames(self):
    return self.class_labels

  def reorder_result(self, result):
    return result

  def __getitem__(self, index):
    out_array = []
    for k in self.loading_key_order:
      loader, transformer = self.data_loader_dict[k]
      v = loader(index)
      if transformer:
        v = transformer(v)
      out_array.append(v)
    return out_array

  def __len__(self):
    return len(self.data_paths)


class VoxelizationDatasetBase(DictDataset, ABC):
  IS_TEMPORAL = False
  CLIP_BOUND = (-1000, -1000, -1000, 1000, 1000, 1000)
  ROTATION_AXIS = None
  NUM_IN_CHANNEL = None
  NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
  IGNORE_LABELS = None  # labels that are not evaluated
  IS_ONLINE_VOXELIZATION = True
  N_ROTATIONS = 1

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               cache=False,
               prefetch_data=False,
               data_root='/',
               ignore_mask=255,
               return_transformation=False,
               rot_aug=False,
               normalize=False,
               normalize_method="sphere",
               **kwargs):
    """
    ignore_mask: label value for ignore class. It will not be used as a class in the loss or evaluation.
    """
    DictDataset.__init__(
        self,
        data_paths,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root)

    self.ignore_mask = ignore_mask
    self.return_transformation = return_transformation
    self.rot_aug = rot_aug
    self.prefetch_data = prefetch_data
    self.normalize = normalize
    self.normalize_method = normalize_method

    if self.prefetch_data:
      self.prefetched_coords, self.prefetched_feats, self.prefetched_labels, \
      self.prefetched_component_ids, self.prefetched_component_names = [], [], [], [], []
      for data_ind in tqdm(range(len(self.data_paths))):
        coords, feats, labels, component_ids, component_names = self.load_ply(data_ind, self.input_feat, self.label_feat)
        if self.normalize:
          coords = t.normalize_coords(coords, self.normalize_method)
        self.prefetched_coords.append(coords)
        self.prefetched_feats.append(feats)
        self.prefetched_labels.append(labels)
        self.prefetched_component_ids.append(component_ids)
        self.prefetched_component_names.append(component_names)
      print("size in bytes of prefetched coords: {}".format(sys.getsizeof(self.prefetched_coords)))
      print("size in bytes of prefetched feats: {}".format(sys.getsizeof(self.prefetched_feats)))
      print("size in bytes of prefetched component_ids: {}".format(sys.getsizeof(self.prefetched_component_ids)))
      print("size in bytes of prefetched component_names: {}".format(sys.getsizeof(self.prefetched_component_names)))
    if self.rot_aug:
      angle = 2 * np.pi / self.N_ROTATIONS
      self.rotation_map = [(data_ind, rot_ind * angle) for data_ind in range(len(data_paths)) for rot_ind in range(self.N_ROTATIONS)]

  def __getitem__(self, index):
    raise NotImplementedError

  def load_ply(self, index, input_feat='rgb', label_feat='label'):

    filepath = self.data_root / self.data_paths[index]
    cmp_filepath = str(filepath).replace(".ply", "_components.json")
    building = os.path.basename(self.data_paths[index])[:-4]

    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.full((coords.shape[0],1), -1)
    if input_feat == 'rgb':
      feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    if input_feat == 'rgba':
      feats = np.array([data['red'], data['green'], data['blue'], data['alpha']], dtype=np.float32).T
    elif input_feat == 'normals' or input_feat == 'xyz_normals':
      feats = np.array([data['nx'], data['ny'], data['nz']], dtype=np.float32).T
    elif input_feat == 'normals_rgb':
      feats = np.array([data['nx'], data['ny'], data['nz'], data['red'], data['green'], data['blue']],
                       dtype=np.float32).T
    elif input_feat == 'normals_rgba':
      feats = np.array([data['nx'], data['ny'], data['nz'], data['red'], data['green'], data['blue'], data['alpha']],
                       dtype=np.float32).T
    elif input_feat == 'occupancy':
      feats = np.ones((coords.shape[0], 1), dtype=np.float32)

    if label_feat in ['rov', 'rnv']:
      labels = np.array(data['rnv'], dtype=np.int32)
      if label_feat == 'rov':
        labels[labels == -1] = 2  # so 0 for none, 1 for ridge, 2 for valley
      else:
        labels[labels == -1] = 1  # so 0 for none, 1 for ridge or valley
    else:
      labels = np.array(data[label_feat], dtype=np.int32)

    component_ids = np.array(data['component'], dtype=np.int32)
    with open(cmp_filepath, "r") as fin:
      component_names = json.load(fin)
    component_names = ["{}_{}".format(building, c) for c in component_names]

    component_ids = [c if c == -1 or "__" not in component_names[c] else c for c in component_ids]
    component_ids = np.array(component_ids, dtype=np.int32)

    return coords, feats, labels, component_ids, component_names

  def __len__(self):
    num_data = len(self.data_paths)
    if self.prefetch_data:
      num_data = len(self.prefetched_coords)
    if self.rot_aug:
     num_data *= self.N_ROTATIONS
    return num_data


class VoxelizationDataset(VoxelizationDatasetBase):
  """This dataset loads RGB point clouds and their labels as a list of points
  and voxelizes the pointcloud with sufficient data augmentation.
  """
  # Voxelization arguments
  VOXEL_SIZE = 0.05  # 5cm

  # Coordinate Augmentation Arguments: Unlike feature augmentation, coordinate
  # augmentation has to be done before voxelization
  SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
  ELASTIC_DISTORT_PARAMS = None

  # MISC.
  PREVOXELIZATION_VOXEL_SIZE = None

  # Augment coords to feats
  AUGMENT_COORDS_TO_FEATS = False

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               data_root='/',
               ignore_label=-1,
               return_transformation=False,
               augment_data=False,
               rot_aug=False,
               config=None,
               **kwargs):

    self.augment_data = augment_data
    self.config = config
    VoxelizationDatasetBase.__init__(
        self,
        data_paths,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        prefetch_data=self.config.prefetch_data,
        data_root=data_root,
        ignore_mask=ignore_label,
        return_transformation=return_transformation,
        rot_aug=rot_aug,
        normalize=config.normalize_coords,
        normalize_method=config.normalize_method,
        input_feat=self.input_feat)

    # Prevoxel transformations
    self.voxelizer = Voxelizer(
        voxel_size=self.VOXEL_SIZE,
        clip_bound=self.CLIP_BOUND,
        use_augmentation=augment_data,
        scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
        rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
        translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        ignore_label=ignore_label)

    # map labels not evaluated to ignore_label
    label_map = {}
    n_used = 0
    for l in range(self.NUM_LABELS):
      if self.IGNORE_LABELS is not None and l in self.IGNORE_LABELS:
        label_map[l] = self.ignore_mask
      else:
        label_map[l] = n_used
        n_used += 1
    label_map[self.ignore_mask] = self.ignore_mask
    self.label_map = label_map
    self.NUM_LABELS -= len(self.IGNORE_LABELS) if self.IGNORE_LABELS is not None else 0

  def _augment_coords_to_feats(self, coords, feats, labels=None):
    if (feats == -1).all():
      feats = coords.copy()
    elif isinstance(coords, np.ndarray):
      feats = np.concatenate((feats, coords), 1)
    else:
      feats = torch.cat((feats, coords), 1)
    return coords, feats, labels

  def convert_mat2cfl(self, mat):
    # Generally, xyz,rgb,label
    return mat[:, :3], mat[:, 3:-1], mat[:, -1]

  def __getitem__(self, index):
    if self.rot_aug:
      if self.config.random_rotation:
        angle = np.random.uniform(self.ROTATION_AUGMENTATION_BOUND[0], self.ROTATION_AUGMENTATION_BOUND[1])
      else:
        index, angle = self.rotation_map[index]
      t.RotationAugmentation.update_angle(angle)
    if self.config.prefetch_data:
      coords = np.copy(self.prefetched_coords[index])
      feats = np.copy(self.prefetched_feats[index])
      labels = np.copy(self.prefetched_labels[index])
      component_ids = np.copy(self.prefetched_component_ids[index])
      component_names = np.copy(self.prefetched_component_names[index])
    else:
      coords, feats, labels, component_ids, component_names = self.load_ply(index, self.input_feat, self.label_feat)
      if self.normalize:
        coords = t.normalize_coords(coords, self.normalize_method)

    # initial_coords_cnt = coords.shape[0]

    # Downsample the pointcloud with finer voxel size before transformation for memory and speed
    if self.PREVOXELIZATION_VOXEL_SIZE is not None:
      inds = ME.utils.sparse_quantize(
          coords / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)
      coords = coords[inds]
      feats = feats[inds]
      labels = labels[inds]

    # Prevoxel transformations
    if self.prevoxel_transform is not None:
      coords, feats, labels = self.prevoxel_transform(coords, feats, labels)

    # Use coordinate features if config is set
    if self.AUGMENT_COORDS_TO_FEATS:
      coords, feats, labels = self._augment_coords_to_feats(coords, feats, labels)

    coords, feats, labels, component_ids, transformation = self.voxelizer.voxelize_both_together(
        coords, feats, labels, component_ids)

    # print("Loaded {} coords/feats/labels & converted into {} (augmented) voxels".format(
    #   initial_coords_cnt, coords.shape[0]))

    # map labels not used for evaluation to ignore_label
    if self.input_transform is not None:
      coords, feats, labels = self.input_transform(coords, feats, labels)
    if self.target_transform is not None:
      coords, feats, labels = self.target_transform(coords, feats, labels)
    if self.IGNORE_LABELS is not None:
      labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

    # c_centers = self.get_component_average(component_ids, coords)
    return_args = [coords, feats, labels, component_ids, component_names]
    if self.return_transformation:
      return_args.append(transformation.astype(np.float32))

    try:
      assert(not np.isnan(coords).any())
    except AssertionError:
      print(self.data_root / self.data_paths[index])
      raise
    try:
      assert(not np.isnan(feats).any())
    except AssertionError:
      print(self.data_root / self.data_paths[index])
      raise
    try:
      assert(not np.isnan(labels).any())
    except AssertionError:
      print(self.data_root / self.data_paths[index])
      raise
    return tuple(return_args)


class TemporalVoxelizationDataset(VoxelizationDataset):

  IS_TEMPORAL = True

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               data_root='/',
               ignore_label=255,
               temporal_dilation=1,
               temporal_numseq=3,
               return_transformation=False,
               augment_data=False,
               config=None,
               **kwargs):
    VoxelizationDataset.__init__(
        self,
        data_paths,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        data_root=data_root,
        ignore_label=ignore_label,
        return_transformation=return_transformation,
        augment_data=augment_data,
        config=config,
        **kwargs)
    self.temporal_dilation = temporal_dilation
    self.temporal_numseq = temporal_numseq
    temporal_window = temporal_dilation * (temporal_numseq - 1) + 1
    self.numels = [len(p) - temporal_window + 1 for p in self.data_paths]
    if any([numel <= 0 for numel in self.numels]):
      raise ValueError('Your temporal window configuration is too wide for '
                       'this dataset. Please change the configuration.')

  def load_world_pointcloud(self, filename):
    raise NotImplementedError

  def __getitem__(self, index):
    for seq_idx, numel in enumerate(self.numels):
      if index >= numel:
        index -= numel
      else:
        break

    numseq = self.temporal_numseq
    if self.augment_data and self.config.temporal_rand_numseq:
      numseq = random.randrange(1, self.temporal_numseq + 1)
    dilations = [self.temporal_dilation for i in range(numseq - 1)]
    if self.augment_data and self.config.temporal_rand_dilation:
      dilations = [random.randrange(1, self.temporal_dilation + 1) for i in range(numseq - 1)]
    files = [self.data_paths[seq_idx][index + sum(dilations[:i])] for i in range(numseq)]

    world_pointclouds = [self.load_world_pointcloud(f) for f in files]
    ptcs, centers = zip(*world_pointclouds)

    # Downsample pointcloud for speed and memory
    if self.PREVOXELIZATION_VOXEL_SIZE is not None:
      new_ptcs = []
      for ptc in ptcs:
        inds = ME.utils.sparse_quantize(
            ptc[:, :3] / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)
        new_ptcs.append(ptc[inds])
      ptcs = new_ptcs

    # Apply prevoxel transformations
    ptcs = [self.prevoxel_transform(ptc) for ptc in ptcs]

    coords, feats, labels = zip(*ptcs)
    outs = self.voxelizer.voxelize_temporal(
        coords, feats, labels, centers=centers, return_transformation=self.return_transformation)

    if self.return_transformation:
      coords_t, feats_t, labels_t, transformation_t = outs
    else:
      coords_t, feats_t, labels_t = outs

    joint_coords = np.vstack([
        np.hstack((coords, np.ones((coords.shape[0], 1)) * i)) for i, coords in enumerate(coords_t)
    ])
    joint_feats = np.vstack(feats_t)
    joint_labels = np.hstack(labels_t)

    # map labels not used for evaluation to ignore_label
    if self.input_transform is not None:
      joint_coords, joint_feats, joint_labels = self.input_transform(joint_coords, joint_feats,
                                                                     joint_labels)
    if self.target_transform is not None:
      joint_coords, joint_feats, joint_labels = self.target_transform(joint_coords, joint_feats,
                                                                      joint_labels)
    if self.IGNORE_LABELS is not None:
      joint_labels = np.array([self.label_map[x] for x in joint_labels], dtype=np.int)

    return_args = [joint_coords, joint_feats, joint_labels]
    if self.return_transformation:
      pointclouds = np.vstack([
          np.hstack((pointcloud[0][:, :6], np.ones((pointcloud[0].shape[0], 1)) * i))
          for i, pointcloud in enumerate(world_pointclouds)
      ])
      transformations = np.vstack(
          [np.hstack((transformation, [i])) for i, transformation in enumerate(transformation_t)])

      return_args.extend([pointclouds.astype(np.float32), transformations.astype(np.float32)])
    return tuple(return_args)

  def __len__(self):
    num_data = sum(self.numels)
    return num_data


def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           num_workers,
                           shuffle,
                           repeat,
                           augment_data,
                           shift,
                           jitter,
                           scale,
                           rot_aug,
                           batch_size,
                           limit_numpoints,
                           input_transform=None,
                           target_transform=None):
  if isinstance(phase, str):
    phase = str2datasetphase_type(phase)

  if config.return_transformation:
    collate_fn = t.cflt_collate_fn_factory(limit_numpoints)
  else:
    collate_fn = t.cfl_collate_fn_factory(limit_numpoints)

  prevoxel_transform_train = []
  if augment_data:
    prevoxel_transform_train.append(t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS))
  if rot_aug:
    prevoxel_transform_train.append(t.RotationAugmentation(True if 'normals' in config.input_feat else False))
  if shift:
    prevoxel_transform_train.append(t.RandomShift(*DatasetClass.SHIFT_PARAMS))
  elif jitter:
    prevoxel_transform_train.append(t.RandomJittering(*DatasetClass.JITTER_AUGMENTATION_BOUND))
  if scale:
    prevoxel_transform_train.append(t.RandomScaling(*DatasetClass.SCALE_AUGMENTATION_BOUND))

  if len(prevoxel_transform_train) > 0:
    prevoxel_transforms = t.Compose(prevoxel_transform_train)
  else:
    prevoxel_transforms = None

  input_transforms = []
  if input_transform is not None:
    input_transforms += input_transform

  if augment_data:
    input_transforms += [
        t.RandomDropout(0.2),
        t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
        t.ChromaticAutoContrast(),
        t.ChromaticTranslation(config.data_aug_color_trans_ratio),
        t.ChromaticJitter(config.data_aug_color_jitter_std),
        # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
    ]

  if len(input_transforms) > 0:
    input_transforms = t.Compose(input_transforms)
  else:
    input_transforms = None

  dataset = DatasetClass(
      config,
      prevoxel_transform=prevoxel_transforms,
      input_transform=input_transforms,
      target_transform=target_transform,
      cache=config.cache_data,
      augment_data=augment_data,
      rot_aug=rot_aug,
      phase=phase)

  data_args = {
      'dataset': dataset,
      'num_workers': num_workers,
      'batch_size': batch_size,
      'collate_fn': collate_fn,
  }

  if repeat:
    data_args['sampler'] = InfSampler(dataset, shuffle)
  else:
    data_args['shuffle'] = shuffle

  data_loader = DataLoader(**data_args)

  return data_loader
