import logging
import os
import tempfile

import numpy as np
import torch
from MinkowskiEngine import SparseTensor
from lib.pc_utils import save_point_cloud
from lib.transforms import normalize_coords
from lib.transforms_extended import get_component_indices_matrix
from lib.utils import Timer, AverageMeter, get_torch_device, precision_at_one, calculate_iou_custom, get_prediction, \
  calculate_shape_iou, calculate_part_iou, calculate_part_iou_custom, update_precision_recall_per_class
from models.modules.non_trainable_layers import get_average_per_component_t


def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               losses,
               scores=None,
               part_iou=None,
               shape_iou=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if scores is None:
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\n".format(loss=losses)
  else:
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
        "Score {acc.val:.3f} (AVG: {acc.avg:.3f})\t" \
        "Part IoU {Part_IoU} Shape IoU {Shape_IoU}\n".format(
            loss=losses, acc=scores, Part_IoU=part_iou,
            Shape_IoU=shape_iou)
  logging.info(debug_str)


def print_the_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               losses,
               scores=None,
               precision_per_class=None,
               recall_per_class=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if scores is None:
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\n".format(loss=losses)
  else:
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
        "Score {acc.val:.3f} (AVG: {acc.avg:.3f})\t".format(loss=losses, acc=scores)
  if precision_per_class is not None:
    debug_str += "\tPrecisions:\n"
    for class_label, p in precision_per_class.items():
      debug_str += f"{class_label}: {p.avg}, "
  if recall_per_class is not None:
    debug_str += "\tRecalls:\n"
    for class_label, r in recall_per_class.items():
      debug_str += f"{class_label}: {r.avg}, "
  logging.info(debug_str)


def test(model, data_loader, criterion, config):
  if criterion.__class__.__name__ == 'CrossEntropyLoss':
    return test_cls_style(model, data_loader, criterion, config)
    # return test_cls(model, data_loader, criterion, config)
  else:
    return test_reconstruction(model, data_loader, criterion, config)

def test_reconstruction(model, data_loader, criterion, config):
  device = get_torch_device(config.is_cuda)
  print("Test_ae with {} device".format(device))
  dataset = data_loader.dataset
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()

  losses = AverageMeter()

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
      sinput, data_time, iter_time, loss, num_sample, soutput, transformation, sqtarg = one_feed_forward_step(config,
                                                                                                      criterion,
                                                                                                      data_loader,
                                                                                                      data_iter,
                                                                                                      device,
                                                                                                      model)
      losses.update(float(loss), num_sample)

      if config.save_prediction or config.test_original_pointcloud:
        save_reconstructed_pointcloud(config, sinput, soutput, transformation, dataset, iteration, save_pred_dir)

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        print_info(iteration, max_iter_unique, data_time, iter_time, losses)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  print_info(iteration, max_iter_unique, data_time, iter_time, losses)

  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  return losses.avg, None, None, None


def test_cls(model, data_loader, criterion, config):
  device = get_torch_device(config.is_cuda)
  print("Test_ae with {} device".format(device))
  dataset = data_loader.dataset
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()

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
      sinput, data_time, iter_time, loss, num_sample, soutput, transformation, sqtarg = one_feed_forward_step(config,
                                                                                                      criterion,
                                                                                                      data_loader,
                                                                                                      data_iter,
                                                                                                      device,
                                                                                                      model)
      losses.update(float(loss), num_sample)
      target = sqtarg.to(device)
      pred = get_prediction(None, soutput.F, None)
      score = precision_at_one(pred, target)
      scores.update(score, num_sample)
      ious[iteration] = calculate_iou_custom(ground=target.cpu().numpy(),  # iou doesnt make sense but ok
                                             prediction=pred.cpu().numpy(),
                                             num_labels=data_loader.dataset.NUM_LABELS,
                                             ignore_label=-1)

      if config.save_prediction or config.test_original_pointcloud:
        raise NotImplementedError()

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        shape_iou = calculate_shape_iou(ious=ious) * 100
        part_iou = calculate_part_iou_custom(ious=ious, num_labels=data_loader.dataset.NUM_LABELS) * 100
        print_info(
          iteration,
          max_iter_unique,
          data_time,
          iter_time,
          losses,
          scores,
          part_iou,
          shape_iou)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  print_info(iteration, max_iter_unique, data_time, iter_time, losses)

  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  return losses.avg, scores.avg, part_iou, shape_iou

def test_cls_style(model, data_loader, criterion, config):
  device = get_torch_device(config.is_cuda)
  print("Test_ae with {} device".format(device))
  dataset = data_loader.dataset
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()

  losses, scores = AverageMeter(), AverageMeter()
  precision_per_class = {c: AverageMeter() for c in dataset.label_map.values() if c != dataset.ignore_mask}
  recall_per_class = {c: AverageMeter() for c in dataset.label_map.values() if c != dataset.ignore_mask}
  inverse_label_map = {v: k for k, v in dataset.label_map.items() if v != dataset.ignore_mask}

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
      sinput, data_time, iter_time, loss, num_sample, soutput, transformation, sqtarg = one_feed_forward_step(config,
                                                                                                      criterion,
                                                                                                      data_loader,
                                                                                                      data_iter,
                                                                                                      device,
                                                                                                      model)
      losses.update(float(loss), num_sample)
      target = sqtarg.to(device)
      pred = get_prediction(None, soutput.F, None)
      score = precision_at_one(pred, target)
      if not np.isnan(score):
        scores.update(score, num_sample)
      else:
        print("WATCH OUT!! : you probably gave sme input buildings with style labels to be ignored")

      update_precision_recall_per_class(pred, target, precision_per_class, recall_per_class)

      if config.save_prediction or config.test_original_pointcloud:
        raise NotImplementedError()

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        ppc = {dataset.CLASS_LABELS[inverse_label_map[c]]: p for c, p in precision_per_class.items()}
        rpc = {dataset.CLASS_LABELS[inverse_label_map[c]]: r for c, r in recall_per_class.items()}
        print_the_info(
          iteration,
          max_iter_unique,
          data_time,
          iter_time,
          losses,
          scores,
          ppc,
          rpc)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  ppc = {inverse_label_map[c]: p for c, p in precision_per_class.items()}
  rpc = {inverse_label_map[c]: r for c, r in recall_per_class.items()}
  print_the_info(iteration, max_iter_unique, data_time, iter_time, losses, scores, ppc, rpc)

  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  return losses.avg, scores.avg

def one_feed_forward_step(config, criterion, data_loader, data_iter, device, model):
  data_timer = Timer()
  iter_timer = Timer()

  data_timer.tic()
  if config.return_transformation:
    coords, input, target, component_ids, component_names, transformation = data_iter.next()
  else:
    coords, input, target, component_ids, component_names = data_iter.next()
    transformation = None
  data_time = data_timer.toc(False)

  # Preprocess input
  iter_timer.tic()
  if config.normalize_color:
    # For BuildNet
    input[:, :3] = input[:, :3] / 255. - config.color_offset

  if not "RNV" in data_loader.dataset.__class__.__name__ and not 'ROV' in data_loader.dataset.__class__.__name__:
    cimat, cnames = get_component_indices_matrix(component_ids, component_names)
    c_i_mat_t = torch.tensor(cimat)
    c_coords_t = get_average_per_component_t(c_i_mat_t, coords)
    c_target_t = get_average_per_component_t(c_i_mat_t, torch.unsqueeze(target, 1), is_target=True)
    sc_indices = SparseTensor(c_i_mat_t, c_coords_t).to(device)
    sc_target = SparseTensor(c_target_t, c_coords_t).to(device)

    sinput = SparseTensor(input, coords).to(device)
    # Feed forward
    inputs = (sinput, sc_indices)
    soutput = model(*inputs)
    iter_time = iter_timer.toc(False)

    squeeze_target = torch.squeeze(sc_target.F.long())
    loss = criterion(soutput.F, squeeze_target)
  else:
    sinput = SparseTensor(input, coords).to(device)
    # Feed forward
    inputs = (sinput, )
    soutput = model(*inputs)
    iter_time = iter_timer.toc(False)

    target = target.long().to(device)
    squeeze_target = torch.squeeze(target.long())
    loss = criterion(soutput.F, squeeze_target)

  num_sample = soutput.shape[0]
  return sinput, data_time, iter_time, loss, num_sample, soutput, transformation, squeeze_target

def save_reconstructed_pointcloud(config, sinput, soutput, transformation, dataset, iteration, save_pred_dir):
  if dataset.IS_ONLINE_VOXELIZATION:
    assert transformation is not None, 'Need transformation matrix.'

  coords_original = sinput.F.detach().cpu().numpy()
  coords_reconstructed = soutput.F.detach().cpu().numpy()

  if config.normalize_y:
    coords_reconstructed = normalize_coords(coords_reconstructed)

  name = os.path.basename(dataset.data_paths[iteration])[:-4]
  filename = os.path.join(save_pred_dir, 'orig_{}.ply'.format(name))
  save_point_cloud(coords_original, filename, binary=False, with_label=False, verbose=False)
  filename = os.path.join(save_pred_dir, 'pred_{}.ply'.format(name))
  save_point_cloud(coords_reconstructed, filename, binary=False, with_label=False, verbose=False)
