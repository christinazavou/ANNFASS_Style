# Change dataloader multiprocess start method to anything not fork
import numpy as np

import torch.multiprocessing as mp
try:
  mp.set_start_method('forkserver')  # Reuse process created
except RuntimeError:
  pass

import os
import sys
import json
import logging
from easydict import EasyDict as edict
from pathlib import Path

# Torch packages
import torch

# Train deps
from config import get_config

import MinkowskiEngine as ME

from lib.train import train
from lib.train_with_component import train as train_with_component
from lib.train_multi_gpu import train as train_multi_gpu
from lib.train_multi_gpu_with_component import train as train_multi_gpu_with_component
from lib.test import test
from lib.test_with_component import test as test_with_component
from lib.utils import get_torch_device, count_parameters, \
    get_with_component_criterion, load_class_weights, get_class_weights
from lib.datasets import load_dataset
from lib.export_features import get_feats

from models import load_model

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def main():
  config = get_config()
  if config.resume:
    max_iter = config.max_iter
    with open(os.path.join(Path(config.resume).parent, 'config.json'), 'r') as fin:
        json_config = json.load(fin)
    json_config['resume'] = config.resume
    # json_config['stylenet_path'] = config.stylenet_path
    # json_config['log_dir'] = config.log_dir
    config = edict(json_config)
    if max_iter > config.max_iter:
      print("Specified max_iter ({}) is higher than in the previous config ({}). "
            "We will use the higher one and update the config file.".format(max_iter, config.max_iter))
      config.max_iter = max_iter
      with open(os.path.join(Path(config.resume).parent, 'config.json'), 'w') as fout:
          json.dump(config, fout, indent=2)

  if config.is_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found")

  if config.multi_gpu:
    # Find maximum number of available devices
    num_devices = torch.cuda.device_count()
    print("found {} devices".format(num_devices))
    num_devices = min(config.max_ngpu, num_devices)
    devices = list(range(num_devices))
    # For copying the final loss back to one GPU
    target_device = devices[0]
    logging.info('===> Multi GPU training')
    for device in devices:
      logging.info('    GPU {}: {}'.format(device, torch.cuda.get_device_name(device)))
    try:
      cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
      logging.info('    CUDA_VISIBLE_DEVICES: {}'.format(cuda_visible_devices))
    except:
      pass
  else:
    target_device = get_torch_device(config.is_cuda)

  logging.info('===> Configurations')
  dconfig = vars(config)
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  DatasetClass = load_dataset(config.dataset)

  if "Component" in str(DatasetClass) or "RNV" in str(DatasetClass) or 'ROV' in str(DatasetClass):
    from lib.dataset_extended import initialize_data_loader
  else:
    from lib.dataset import initialize_data_loader

  if config.test_original_pointcloud:
    if not DatasetClass.IS_FULL_POINTCLOUD_EVAL:
      raise ValueError('This dataset does not support full pointcloud evaluation.')

  if config.evaluate_original_pointcloud:
    if not config.return_transformation:
      raise ValueError('Pointcloud evaluation requires config.return_transformation=true.')

  if config.export_feat:
    get_feats(DatasetClass, config)
    exit(0)

  logging.info('===> Initializing dataloader')
  if config.is_train:
    batch_size = config.batch_size
    if config.multi_gpu:
      # Devide batch size into multiple gpus
      assert(config.batch_size % num_devices == 0)
      batch_size = int(config.batch_size / num_devices)
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
        batch_size=batch_size,
        limit_numpoints=config.train_limit_numpoints)

    val_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        num_workers=config.num_val_workers,
        phase=config.val_phase,
        augment_data=False,
        shift=False,
        jitter=False,
        rot_aug=False,
        scale=False,
        shuffle=False,  # no need to be shuffled, and if shuffle is true it wont allow empty dataset
        repeat=False,
        batch_size=config.val_batch_size,
        limit_numpoints=False)
    if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
      num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
    else:
      num_in_channel = 3  # RGB color or point normals

    num_labels = train_data_loader.dataset.NUM_LABELS
  else:
    test_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        num_workers=config.num_workers,
        phase=config.test_phase,
        augment_data=False,
        shift=False,
        jitter=False,
        rot_aug=False,
        scale=False,
        shuffle=False,
        repeat=False,
        batch_size=config.test_batch_size,
        limit_numpoints=False)
    if test_data_loader.dataset.NUM_IN_CHANNEL is not None:
      num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
    else:
      num_in_channel = 3  # RGB color or point normals

    num_labels = test_data_loader.dataset.NUM_LABELS

  logging.info('===> Building model')
  NetClass = load_model(config.model)
  model = NetClass(num_in_channel, num_labels, config)
  logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__, count_parameters(model)))
  logging.info(model)
  model = model.to(target_device)
  if config.multi_gpu:
    # Synchronized batch norm
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

  test_iter = 0
  # Load weights if specified by the parameter.
  if config.weights.lower() != 'none':
    logging.info('===> Loading weights: ' + config.weights)
    state = torch.load(config.weights)
    model.load_state_dict(state['state_dict'])
    if 'iteration' in state:
        test_iter = state['iteration']

  if config.ps_weights.lower() != 'none':
      model.load_from_ps(config.ps_weights)

  if config.is_train:
    assert len(train_data_loader) != 0, "Can't train with 0 samples"
    if 'StyleCls' in str(model.__class__) or "RNV" in str(train_data_loader.dataset) or 'ROV' in str(train_data_loader.dataset):
        if config.multi_gpu:
          train_multi_gpu_with_component(model, train_data_loader, val_data_loader, devices, config)
        else:
          train_with_component(model, train_data_loader, val_data_loader, config)
    else:
        if config.multi_gpu:
          train_multi_gpu(model, train_data_loader, val_data_loader, devices, config)
        else:
          train(model, train_data_loader, val_data_loader, config)
  else:
    if 'StyleCls' in str(model.__class__) or "RNV" in str(test_data_loader.dataset) or 'ROV' in str(test_data_loader.dataset):
        if config.class_weights != "":
            class_weights = load_class_weights(config.class_weights)
        else:
            if config.class_balanced_loss or config.inv_freq_class_weight:
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
                    batch_size=1,
                    limit_numpoints=config.train_limit_numpoints)
                class_weights = get_class_weights(config, train_data_loader.dataset)
            else:
                class_weights = None
        criterion = get_with_component_criterion(config, test_data_loader.dataset, class_weights)
        test_with_component(model, test_data_loader, criterion, config)
    else:
        weights = None
        if config.weighted_cross_entropy or config.weighted_focal_loss:
            raise Exception("OPS")
        elif config.class_balanced_loss:
            raise Exception("OPS")
        loss, score, part_iou, shape_iou = test(model, test_data_loader, config, weights=weights)
        logging.info("Test split Part IOU: {:.3f} at iter {}".format(part_iou, test_iter))
        logging.info("Test split Shape IOU: {:.3f} at iter {}".format(shape_iou, test_iter))
        logging.info("Test split Loss: {:.3f} at iter {}".format(loss, test_iter))
        logging.info("Test Score: {:.3f} at iter {}".format(score, test_iter))


if __name__ == '__main__':
  __spec__ = None

  main()
