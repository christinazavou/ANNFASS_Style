import errno
import json
import logging
import os
import time
import math

import numpy as np
import torch
from torch import nn
from lib.pc_utils import colorize_pointcloud, save_point_cloud
from models.modules.non_trainable_layers import get_average_per_component_t
import lib.transforms_extended as t


def load_state_with_same_shape(model, weights):
  model_state = model.state_dict()
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights


def checkpoint(model, optimizer, epoch, iteration, config, best_val_part_iou=None, best_val_part_iou_iter=None,
               best_val_shape_iou=None, best_val_shape_iou_iter=None, best_val_loss=None,  best_val_loss_iter=None,
               best_val_acc=None, best_val_acc_iter=None, postfix=None):
  mkdir_p(config.log_dir)
  if config.overwrite_weights:
    if postfix is not None:
      filename = f"checkpoint_{config.model}{postfix}.pth"
    else:
      filename = f"checkpoint_{config.model}.pth"
  else:
    filename = f"checkpoint_{config.model}_iter_{iteration}.pth"
  checkpoint_file = config.log_dir + '/' + filename
  state = {
      'iteration': iteration,
      'epoch': epoch,
      'arch': config.model,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
  }
  if best_val_part_iou is not None:
    state['best_val_part_iou'] = best_val_part_iou
    state['best_val_part_iou_iter'] = best_val_part_iou_iter
  if best_val_shape_iou is not None:
    state['best_val_shape_iou'] = best_val_shape_iou
    state['best_val_shape_iou_iter'] = best_val_shape_iou_iter
  if best_val_loss  is not None:
    state['best_val_loss'] = best_val_loss
    state['best_val_loss_iter'] = best_val_loss_iter
  if best_val_loss  is not None:
    state['best_val_acc'] = best_val_acc
    state['best_val_acc_iter'] = best_val_acc_iter
  with open(config.log_dir + '/config.json', 'w') as fout:
    json.dump(vars(config), fout, indent=4)
  torch.save(state, checkpoint_file)
  logging.info(f"Checkpoint saved to {checkpoint_file}")

  if postfix is None:
    # Delete symlink if it exists
    if os.path.exists(f'{config.log_dir}/weights.pth'):
      os.remove(f'{config.log_dir}/weights.pth')
    # Create symlink
    os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')


def precision_at_one(pred, target, ignore_label=255):
  """Computes the precision@k for the specified values of k"""
  # batch_size = target.size(0) * target.size(1) * target.size(2)
  pred = pred.view(1, -1)
  target = target.view(1, -1)
  correct = pred.eq(target)
  correct = correct[target != ignore_label]
  correct = correct.view(-1)
  if correct.nelement():
    return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
  else:
    return float('nan')


def update_precision_recall_per_class(pred, target, precision_per_class, recall_per_class):
  for class_label in precision_per_class:
    class_predictions = (pred == class_label)
    class_labels = (target == class_label)
    num_correct_class_predictions = torch.sum(torch.bitwise_and(class_predictions, class_labels))
    if torch.sum(class_predictions) != 0:
      precision = torch.true_divide(num_correct_class_predictions, torch.sum(class_predictions))
      precision_per_class[class_label].update(precision, 1)
    if torch.sum(class_labels) != 0:
      recall = torch.true_divide(num_correct_class_predictions, torch.sum(class_labels))
      recall_per_class[class_label].update(recall, 1)


def calculate_iou(ground, prediction, num_labels):
  """
    Calculate point IOU
  :param ground: N x 1, numpy.ndarray(int)
  :param prediction: N x 1, numpy.ndarray(int)
  :param num_labels: int
  :return:
    metrics: dict: {
                    "label_iou": dict{label: iou (float)},
                    "intersection": dict{label: intersection (float)},
                    "union": dict{label: union (float)
                   }
  """

  label_iou, intersection, union = {}, {}, {}
  # Ignore undetermined
  prediction = np.copy(prediction)
  prediction[ground == 0] = 0

  for i in range(1, num_labels):
    # Calculate intersection and union for ground truth and predicted labels
    intersection_i = np.sum((ground == i) & (prediction== i))
    union_i = np.sum((ground == i) | (prediction == i))

    # If label i is present either on the gt or the pred set
    if union_i > 0:
      intersection[i] = float(intersection_i)
      union[i] = float(union_i)
      label_iou[i] = intersection[i] / union[i]

  metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

  return metrics


def calculate_iou_custom(ground, prediction, num_labels, ignore_label=255):

  label_iou, intersection, union = {}, {}, {}
  # Ignore undetermined
  prediction = np.copy(prediction)
  prediction[ground == 0] = 0

  for i in range(0, num_labels):
    if i == ignore_label:
      continue
    # Calculate intersection and union for ground truth and predicted labels
    intersection_i = np.sum((ground == i) & (prediction== i))
    union_i = np.sum((ground == i) | (prediction == i))

    # If label i is present either on the gt or the pred set
    if union_i > 0:
      intersection[i] = float(intersection_i)
      union[i] = float(union_i)
      label_iou[i] = intersection[i] / union[i]

  metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

  return metrics


def calculate_shape_iou(ious):
  """
    Average label IOU and calculate overall shape IOU
  :param ious: dict: {
                      <model_name> : {
                                      "label_iou": dict{label: iou (float)},
                                      "intersection": dict{label: intersection (float)},
                                      "union": dict{label: union (float)
                                     }
                     }
  :return:
    avg_shape_iou: float
  """

  shape_iou = {}

  for model_name, metrics in ious.items():
    # Average label iou per shape
    L_s = len(metrics["label_iou"])
    shape_iou[model_name] = np.nan_to_num(np.sum([v for v in metrics["label_iou"].values()]) / float(L_s))

  # Dataset avg shape iou
  avg_shape_iou = np.sum([v for v in shape_iou.values()]) / float(len(ious))

  return avg_shape_iou


def calculate_part_iou(ious, num_labels):
  """
    Average intersection/union and calculate overall part IOU
  :param ious: dict: {
                      <model_name> : {
                                      "label_iou": dict{label: iou (float)},
                                      "intersection": dict{label: intersection (float)},
                                      "union": dict{label: union (float)
                                     }
                     }
  :param num_labels: int
  :return:
    avg_part_iou: float
  """

  intersection = {i: 0.0 for i in range(1, num_labels)}
  union = {i: 0.0 for i in range(1, num_labels)}

  for model_name, metrics in ious.items():
    for label in metrics["intersection"].keys():
      # Accumulate intersection and union for each label across all shapes
      intersection[label] += metrics["intersection"][label]
      union[label] += metrics["union"][label]

  # Calculate part IOU for each label
  part_iou = {}
  for key in range(1, num_labels):
    try:
      part_iou[key] = intersection[key] / union[key]
    except ZeroDivisionError:
      part_iou[key] = 0.0
  # Avg part IOU
  avg_part_iou = np.sum([v for v in part_iou.values()]) / float(num_labels - 1)

  return avg_part_iou


def calculate_part_iou_custom(ious, num_labels):
  """
    Average intersection/union and calculate overall part IOU
  :param ious: dict: {
                      <model_name> : {
                                      "label_iou": dict{label: iou (float)},
                                      "intersection": dict{label: intersection (float)},
                                      "union": dict{label: union (float)
                                     }
                     }
  :param num_labels: int
  :return:
    avg_part_iou: float
  """

  intersection = {i: 0.0 for i in range(0, num_labels)}
  union = {i: 0.0 for i in range(0, num_labels)}

  for model_name, metrics in ious.items():
    for label in metrics["intersection"].keys():
      # Accumulate intersection and union for each label across all shapes
      intersection[label] += metrics["intersection"][label]
      union[label] += metrics["union"][label]

  # Calculate part IOU for each label
  part_iou = {}
  for key in range(0, num_labels):
    try:
      part_iou[key] = intersection[key] / union[key]
    except ZeroDivisionError:
      part_iou[key] = 0.0
  # Avg part IOU
  avg_part_iou = np.sum([v for v in part_iou.values()]) / float(num_labels)

  return avg_part_iou


def fast_hist(pred, label, n):
  k = (label >= 0) & (label < n)
  return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class WithTimer(object):
  """Timer for with statement."""

  def __init__(self, name=None):
    self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    out_str = 'Elapsed: %s' % (time.time() - self.tstart)
    if self.name:
      logging.info('[{self.name}]')
    logging.info(out_str)


class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0
    self.averate_time = 0

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff


class ExpTimer(Timer):
  """ Exponential Moving Average Timer """

  def __init__(self, alpha=0.5):
    super(ExpTimer, self).__init__()
    self.alpha = alpha

  def toc(self):
    self.diff = time.time() - self.start_time
    self.average_time = self.alpha * self.diff + \
        (1 - self.alpha) * self.average_time
    return self.average_time


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def debug_on():
  import sys
  import pdb
  import functools
  import traceback

  def decorator(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      try:
        return f(*args, **kwargs)
      except Exception:
        info = sys.exc_info()
        traceback.print_exception(*info)
        pdb.post_mortem(info[2])

    return wrapper

  return decorator


def get_prediction(dataset, output, target):
  return output.max(1)[1]


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
  return torch.device('cuda' if is_cuda else 'cpu')


class HashTimeBatch(object):

  def __init__(self, prime=5279):
    self.prime = prime

  def __call__(self, time, batch):
    return self.hash(time, batch)

  def hash(self, time, batch):
    return self.prime * batch + time

  def dehash(self, key):
    time = key % self.prime
    batch = key / self.prime
    return time, batch


def save_rotation_pred(iteration, pred, dataset, save_pred_dir):
  """Save prediction results in original pointcloud scale."""
  decode_label_map = {}
  for k, v in dataset.label_map.items():
    decode_label_map[v] = k
  pred = np.array([decode_label_map[x] for x in pred], dtype=np.int)
  out_rotation_txt = dataset.get_output_id(iteration) + '.txt'
  out_rotation_path = save_pred_dir + '/' + out_rotation_txt
  np.savetxt(out_rotation_path, pred, fmt='%i')


def save_predictions(coords, pred, transformation, dataset, iteration, save_pred_dir):
  """Save prediction results in original pointcloud scale"""
  if dataset.IS_ONLINE_VOXELIZATION:
    assert transformation is not None, 'Need transformation matrix.'

    # Calculate original coordinates.
    coords_original = coords[:, 1:].numpy()
    if dataset.IS_ONLINE_VOXELIZATION:
      # Undo voxelizer transformation.
      curr_transformation = transformation[0][:16].numpy().reshape(4, 4)
      xyz = np.hstack((coords_original, np.ones((coords_original.shape[0], 1))))
      orig_coords = (np.linalg.inv(curr_transformation) @ xyz.T).T
    else:
      orig_coords = coords_original
    # Undo ignore label masking to fit original dataset label.
    orig_pred = pred.detach().cpu().numpy()
    if dataset.IGNORE_LABELS:
      decode_label_map = {}
      for k, v in dataset.label_map.items():
        decode_label_map[v] = k
      orig_pred = np.array([decode_label_map[x] for x in orig_pred], dtype=np.int)
    # Determine full path of the destination.
    full_pred = np.hstack((orig_coords[:, :3], np.expand_dims(orig_pred, 1)))

    name = os.path.basename(dataset.data_paths[iteration])[:-4]
    # Save final prediction as npy format.
    np.save(os.path.join(save_pred_dir, 'pred_{}.npy'.format(name)), full_pred)


def save_output_features(coords, out_feat, pred, transformation, dataset, iteration, save_pred_dir):
  """Save output features results in original pointcloud scale"""
  if dataset.IS_ONLINE_VOXELIZATION:
    assert transformation is not None, 'Need transformation matrix.'

  # Calculate original coordinates.
  coords_original = coords[:, 1:].numpy()
  if dataset.IS_ONLINE_VOXELIZATION:
    # Undo voxelizer transformation.
    curr_transformation = transformation[0][:16].numpy().reshape(4, 4)
    xyz = np.hstack((coords_original, np.ones((coords_original.shape[0], 1))))
    orig_coords = (np.linalg.inv(curr_transformation) @ xyz.T).T
  else:
    orig_coords = coords_original
  # Undo ignore label masking to fit original dataset label.
  orig_pred = pred.detach().cpu().numpy()
  if dataset.IGNORE_LABELS:
    decode_label_map = {}
    for k, v in dataset.label_map.items():
      decode_label_map[v] = k
    orig_pred = np.array([decode_label_map[x] for x in orig_pred], dtype=np.int)
  out_feat = out_feat.detach().cpu().numpy()
  # Determine full path of the destination.
  full_pred = np.hstack((orig_coords[:, :3], out_feat, np.expand_dims(orig_pred, 1)))
  filename = 'pred_%04d.npy' % (iteration)
  # Save final prediction as npy format.
  np.save(os.path.join(save_pred_dir, filename), full_pred)


def visualize_results(coords, input, target, upsampled_pred, config, iteration):
  # Get filter for valid predictions in the first batch.
  target_batch = coords[:, 3].numpy() == 0
  input_xyz = coords[:, :3].numpy()
  target_valid = target.numpy() != 255
  target_pred = np.logical_and(target_batch, target_valid)
  target_nonpred = np.logical_and(target_batch, ~target_valid)
  ptc_nonpred = np.hstack((input_xyz[target_nonpred], np.zeros((np.sum(target_nonpred), 3))))
  # Unwrap file index if tested with rotation.
  file_iter = iteration
  if config.test_rotation >= 1:
    file_iter = iteration // config.test_rotation
  # Create directory to save visualization results.
  os.makedirs(config.visualize_path, exist_ok=True)
  # Label visualization in RGB.
  xyzlabel = colorize_pointcloud(input_xyz[target_pred], upsampled_pred[target_pred])
  xyzlabel = np.vstack((xyzlabel, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'pred', '%04d.ply' % file_iter])
  save_point_cloud(xyzlabel, os.path.join(config.visualize_path, filename), verbose=False)
  # RGB input values visualization.
  xyzrgb = np.hstack((input_xyz[target_batch], input[:, :3].cpu().numpy()[target_batch]))
  filename = '_'.join([config.dataset, config.model, 'rgb', '%04d.ply' % file_iter])
  save_point_cloud(xyzrgb, os.path.join(config.visualize_path, filename), verbose=False)
  # Ground-truth visualization in RGB.
  xyzgt = colorize_pointcloud(input_xyz[target_pred], target.numpy()[target_pred])
  xyzgt = np.vstack((xyzgt, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'gt', '%04d.ply' % file_iter])
  save_point_cloud(xyzgt, os.path.join(config.visualize_path, filename), verbose=False)


def permute_pointcloud(input_coords, pointcloud, transformation, label_map,
                       voxel_output, voxel_pred):
  """Get permutation from pointcloud to input voxel coords."""
  def _hash_coords(coords, coords_min, coords_dim):
    return np.ravel_multi_index((coords - coords_min).T, coords_dim)
  # Validate input.
  input_batch_size = input_coords[:, -1].max().item()
  pointcloud_batch_size = pointcloud[:, -1].max().int().item()
  transformation_batch_size = transformation[:, -1].max().int().item()
  assert input_batch_size == pointcloud_batch_size == transformation_batch_size
  pointcloud_permutation, pointcloud_target = [], []
  # Process each batch.
  for i in range(input_batch_size + 1):
    # Filter batch from the data.
    input_coords_mask_b = input_coords[:, -1] == i
    input_coords_b = (input_coords[input_coords_mask_b])[:, :-1].numpy()
    pointcloud_b = pointcloud[pointcloud[:, -1] == i, :-1].numpy()
    transformation_b = transformation[i, :-1].reshape(4, 4).numpy()
    # Transform original pointcloud to voxel space.
    original_coords1 = np.hstack((pointcloud_b[:, :3], np.ones((pointcloud_b.shape[0], 1))))
    original_vcoords = np.floor(original_coords1 @ transformation_b.T)[:, :3].astype(int)
    # Hash input and voxel coordinates to flat coordinate.
    vcoords_all = np.vstack((input_coords_b, original_vcoords))
    vcoords_min = vcoords_all.min(0)
    vcoords_dims = vcoords_all.max(0) - vcoords_all.min(0) + 1
    input_coords_key = _hash_coords(input_coords_b, vcoords_min, vcoords_dims)
    original_vcoords_key = _hash_coords(original_vcoords, vcoords_min, vcoords_dims)
    # Query voxel predictions from original pointcloud.
    key_to_idx = dict(zip(input_coords_key, range(len(input_coords_key))))
    pointcloud_permutation.append(
        np.array([key_to_idx.get(i, -1) for i in original_vcoords_key]))
    pointcloud_target.append(pointcloud_b[:, -1].astype(int))
  pointcloud_permutation = np.concatenate(pointcloud_permutation)
  # Prepare pointcloud permutation array.
  pointcloud_permutation = torch.from_numpy(pointcloud_permutation)
  permutation_mask = pointcloud_permutation >= 0
  permutation_valid = pointcloud_permutation[permutation_mask]
  # Permuate voxel output to pointcloud.
  pointcloud_output = torch.zeros(pointcloud.shape[0], voxel_output.shape[1]).to(voxel_output)
  pointcloud_output[permutation_mask] = voxel_output[permutation_valid]
  # Permuate voxel prediction to pointcloud.
  # NOTE: Invalid points (points found in pointcloud but not in the voxel) are mapped to 0.
  pointcloud_pred = torch.ones(pointcloud.shape[0]).int().to(voxel_pred) * 0
  pointcloud_pred[permutation_mask] = voxel_pred[permutation_valid]
  # Map pointcloud target to respect dataset IGNORE_LABELS
  pointcloud_target = torch.from_numpy(
      np.array([label_map[i] for i in np.concatenate(pointcloud_target)])).int()
  return pointcloud_output, pointcloud_pred, pointcloud_target



def get_with_component_criterion(config, dataset, class_weights):
  if 'RNV' in dataset.__class__.__name__ or 'ROV' in dataset.__class__.__name__:
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
  else:
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=config.ignore_label)
  return criterion


def inverse_freq(frequencies):  # pass ony the classes that are considered. not ignored ones
  max_freq = max(frequencies.values())
  weights = {l: 1 + math.log10(max_freq / f) if f > 0 else 0 for l, f in frequencies.items()}
  min_w = min([w for l, w in weights.items() if w > 0])
  max_w = max([w for l, w in weights.items() if w > 0])
  weights = {l: (w - min_w) / (max_w - min_w) + 1 if w >0 else 0 for l, w in weights.items()}
  return weights


def class_balanced(frequencies, b=0.9):  # pass ony the classes that are considered. not ignored ones
  effective_num = {l: 1.0 - np.power(b, f) if f > 0 else 0 for l, f in frequencies.items()}
  weights = {l: (1.0 - b) / e if e > 0 else 0 for l, e in effective_num.items()}
  len_f = len(frequencies) - 1
  weights = {l: w / np.sum(list(weights.values())) * len_f if w > 0 else 0 for l, w in weights.items()}
  return weights


def get_class_counts(dataset):
  label_map_inverse = {v: l for l, v in dataset.label_map.items()}  # ignore_mask can be anything
  label_map_inverse[dataset.ignore_mask] = dataset.ignore_mask
  final_label_counts = {}
  for model_label in label_map_inverse:
    final_label_counts[model_label] = 0
  str_label_counts = {'ignore': 0}
  for i, str_label in enumerate(dataset.CLASS_LABELS):
    str_label_counts[str_label] = 0
  for i in range(len(dataset)):
    # not sparse-quantized
    coords, input, style_labels, component_ids, component_names = dataset.__getitem__(i)
    cimat, cnames = t.get_component_indices_matrix(component_ids, component_names)
    c_i_mat_t = torch.tensor(cimat)
    style_labels = torch.from_numpy(style_labels).unsqueeze(1)
    style_label_per_component = get_average_per_component_t(c_i_mat_t, style_labels)
    for label in style_label_per_component:
      if label == dataset.ignore_mask:
        final_label_counts[dataset.ignore_mask] += 1
        str_label_counts['ignore'] += 1
      elif int(label) in dataset.IGNORE_LABELS:
        final_label_counts[dataset.ignore_mask] += 1
        str_label_counts[dataset.CLASS_LABELS[int(label)]] += 1
      else:
        model_label = dataset.label_map[int(label)]
        final_label_counts[model_label] += 1
        str_label_counts[dataset.CLASS_LABELS[int(label)]] += 1
  print(f"str_class_counts: {str_label_counts}")
  print(f"final_class_counts: {final_label_counts}")
  return final_label_counts, str_label_counts


def get_class_frequencies(counts, ignore_label):
  frequencies = {i: 0 for i in counts}
  total_counts = np.sum(list(counts.values()))
  for label, count in counts.items():
    if label != ignore_label:
      frequencies[label] = count / total_counts
  return frequencies


def get_class_weights(config, dataset):
  if config.class_balanced_loss or config.inv_freq_class_weight:
    final_class_counts, _ = get_class_counts(dataset)
    class_frequencies = get_class_frequencies(final_class_counts, config.ignore_label)
    print(f"class frequencies: {class_frequencies}")
    del class_frequencies[config.ignore_label]
    if config.class_balanced_loss:
      class_weights = class_balanced(class_frequencies, config.class_balanced_beta)
    else:
      class_weights = inverse_freq(class_frequencies)
    print(f"class_weights: {class_weights}")
    class_weights = torch.from_numpy(np.array(list(class_weights.values()))).float().cuda()
  else:
    class_weights = None
  return class_weights


def load_class_weights(npy_file):
  weights = np.load(npy_file)
  if weights is np.NAN:
    return None
  else:
    weights = torch.from_numpy(weights).float().cuda()
  return weights


def save_class_weights(weights, npy_file):
  if weights is None:
    np.save(npy_file, np.NAN)
  else:
    np.save(npy_file, weights.detach().cpu().numpy())
