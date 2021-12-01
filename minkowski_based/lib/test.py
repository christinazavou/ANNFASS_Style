import logging
import os
import tempfile

import torch
import torch.nn as nn

from lib.utils import Timer, AverageMeter, calculate_iou, calculate_shape_iou, calculate_part_iou, precision_at_one, \
  get_prediction, get_torch_device, save_output_features
from lib.focal_loss import FocalLoss

from MinkowskiEngine import SparseTensor


def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               has_gt=False,
               losses=None,
               scores=None,
               part_iou=None,
               shape_iou=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if has_gt:
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
        "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
        "Part IoU {Part_IoU:.3f} Shape IoU {Shape_IoU:.3f}\n".format(
            loss=losses, top1=scores, Part_IoU=part_iou,
            Shape_IoU=shape_iou)


def test(model, data_loader, config, has_gt=True, weights=None):
  device = get_torch_device(config.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  if config.weighted_cross_entropy or config.class_balanced_loss:
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label, weight=weights)
  elif config.focal_loss or config.weighted_focal_loss:
    criterion = FocalLoss(alpha=weights, gamma=2.0, ignore_index=config.ignore_label)
  else:
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
  losses, scores, ious, shape_iou, part_iou = AverageMeter(), AverageMeter(), {}, 0.0, 0.0

  logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter

  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()

  if config.save_prediction:
    save_pred_dir = config.save_pred_dir
    os.makedirs(save_pred_dir, exist_ok=True)
  else:
    save_pred_dir = tempfile.mkdtemp()
  if os.listdir(save_pred_dir):
    raise ValueError(f'Directory {save_pred_dir} not empty. '
                     'Please remove the existing prediction.')

  with torch.no_grad():
    for iteration in range(max_iter):
      data_timer.tic()
      if config.return_transformation:
        coords, input, target, transformation = data_iter.next()
      else:
        coords, input, target = data_iter.next()
        transformation = None
      data_time = data_timer.toc(False)

      # Preprocess input
      iter_timer.tic()

      if config.normalize_color:
        # For BuildNet
        input[:, :3] = input[:, :3] / 255. - config.color_offset
      sinput = SparseTensor(input, coords).to(device)

      # Feed forward
      inputs = (sinput,)
      soutput = model(*inputs)
      output = soutput.F

      pred = get_prediction(dataset, output, target).int()
      iter_time = iter_timer.toc(False)

      if config.save_prediction:
        save_output_features(coords, output, pred, transformation, dataset, iteration, save_pred_dir)

      if has_gt:
        num_sample = target.shape[0]
        target = target.to(device)
        cross_ent = criterion(output, target.long())
        losses.update(float(cross_ent), num_sample)
        scores.update(precision_at_one(pred, target), num_sample)
        ious[iteration] = calculate_iou(ground=target.cpu().numpy(), prediction=pred.cpu().numpy(), num_labels=num_labels)

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        shape_iou = calculate_shape_iou(ious=ious) * 100
        part_iou = calculate_part_iou(ious=ious, num_labels=num_labels) * 100
        print_info(
          iteration,
          max_iter_unique,
          data_time,
          iter_time,
          has_gt,
          losses,
          scores,
          part_iou,
          shape_iou)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  shape_iou = calculate_shape_iou(ious=ious) * 100
  part_iou = calculate_part_iou(ious=ious, num_labels=num_labels) * 100
  print_info(
    iteration,
    max_iter_unique,
    data_time,
    iter_time,
    has_gt,
    losses,
    scores,
    part_iou,
    shape_iou)

  if config.test_original_pointcloud:
    logging.info('===> Start testing on original pointcloud space.')
    dataset.test_pointcloud(save_pred_dir)

  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  return losses.avg, scores.avg, part_iou, shape_iou
