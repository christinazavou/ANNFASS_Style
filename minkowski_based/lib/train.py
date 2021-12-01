import logging
import os.path as osp
import numpy as np

import torch
from torch import nn

from tensorboardX import SummaryWriter

from lib.test import test
from lib.utils import checkpoint, precision_at_one, \
    Timer, AverageMeter, get_prediction, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler
from lib.focal_loss import FocalLoss

from MinkowskiEngine import SparseTensor


def validate(model, val_data_loader, writer, curr_iter, config, weights=None):
  loss, score, part_iou, shape_iou = test(model, val_data_loader, config, weights=weights)
  writer.add_scalar('validation/PartIoU', part_iou, curr_iter)
  writer.add_scalar('validation/ShapeIoU', shape_iou, curr_iter)
  writer.add_scalar('validation/loss', loss, curr_iter)
  writer.add_scalar('validation/precision_at_1', score, curr_iter)

  return loss, score, part_iou, shape_iou


def train(model, data_loader, val_data_loader, config, transform_data_fn=None):
  device = get_torch_device(config.is_cuda)
  # Set up the train flag for batch normalization
  model.train()

  # Configuration
  writer = SummaryWriter(log_dir=config.log_dir)
  data_timer, iter_timer = Timer(), Timer()
  data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
  losses, scores = AverageMeter(), AverageMeter()

  optimizer = initialize_optimizer(model.parameters(), config)
  scheduler = initialize_scheduler(optimizer, config)
  weights = None
  if config.weighted_cross_entropy or config.weighted_focal_loss:
    # Re-weight the loss function by assigning relatively higher costs to examples from minor classes
    weights = []
    for id in range(1, data_loader.dataset.NUM_LABELS+1):
      weights.append(data_loader.dataset.NORM_FREQ[id])
    weights = np.asarray(weights)
    weights = 1 + np.log10(weights)
    min_weight = np.amin(weights)
    max_weight = np.amax(weights)
    weights = (weights - min_weight) / (max_weight - min_weight) + 1
    weights = torch.from_numpy(weights).float()
    weights = weights.cuda()
  elif config.class_balanced_loss:
    # Use Class-balanced loss re-weight method - inverse effective number of samples for each class
    # https://github.com/richardaecn/class-balanced-loss/blob/1d7857208a2abc03d84e35a9d5383af8225d4b4d/src/cifar_main.py#L425-L430
    freq = []
    for id in range(1, data_loader.dataset.NUM_LABELS + 1):
      freq.append(data_loader.dataset.FREQ[id])
    freq = np.asarray(freq)
    effective_num = 1.0 - np.power(config.class_balanced_beta, freq)
    weights = (1.0 - config.class_balanced_beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * data_loader.dataset.NUM_LABELS
    assert(np.isclose(np.sum(weights), data_loader.dataset.NUM_LABELS))
    weights = torch.from_numpy(weights).float()
    weights = weights.cuda()
  if config.weighted_cross_entropy or config.class_balanced_loss:
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label, weight=weights)
  elif config.focal_loss or config.weighted_focal_loss:
    criterion = FocalLoss(alpha=weights, gamma=2.0, ignore_index=config.ignore_label)
  else:
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

  writer = SummaryWriter(log_dir=config.log_dir)

  # Train the network
  logging.info('===> Start training')
  best_val_part_iou, best_val_part_iou_iter, best_val_shape_iou, best_val_shape_iou_iter = 0, 0, 0 ,0
  best_val_loss, best_val_loss_iter, best_val_acc, best_val_acc_iter = np.Inf, 0, 0, 0
  curr_iter, epoch, is_training = 1, 1, True

  if config.resume:
    checkpoint_fn = config.resume + '/weights.pth'
    if osp.isfile(checkpoint_fn):
      logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn)
      curr_iter = state['iteration'] + 1
      epoch = state['epoch']
      model.load_state_dict(state['state_dict'])
      if config.resume_optimizer:
        scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
        optimizer.load_state_dict(state['optimizer'])
      if 'best_val_part_iou' in state:
        best_val_part_iou = state['best_val_part_iou']
        best_val_part_iou_iter = state['best_val_part_iou_iter']
      if 'best_val_shape_iou' in state:
        best_val_shape_iou = state['best_val_shape_iou']
        best_val_shape_iou_iter = state['best_val_shape_iou_iter']
      if 'best_val_loss' in state:
        best_val_loss = state['best_val_loss']
        best_val_loss_iter = state['best_val_loss_iter']
      if 'best_val_acc' in state:
        best_val_acc = state['best_val_acc']
        best_val_acc_iter = state['best_val_acc_iter']
      logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

  data_iter = data_loader.__iter__()
  torch.autograd.set_detect_anomaly(True)
  while is_training:
    for iteration in range(len(data_loader) // config.iter_size):
      optimizer.zero_grad()
      data_time, batch_loss = 0, 0
      iter_timer.tic()

      for sub_iter in range(config.iter_size):
        # Get training data
        data_timer.tic()
        coords, input, target = data_iter.next()

        # Preprocess input
        if config.normalize_color:
          # For BuildNet
          input[:, :3] = input[:, :3] / 255. - config.color_offset
        sinput = SparseTensor(input, coords).to(device)

        data_time += data_timer.toc(False)

        # Feed forward
        inputs = (sinput,)
        soutput = model(*inputs)
        # The output of the network is not sorted
        target = target.long().to(device)

        loss = criterion(soutput.F, target.long())

        # Compute and accumulate gradient
        loss /= config.iter_size
        batch_loss += loss.item()
        loss.backward()
        torch.cuda.empty_cache()

      # Update number of steps
      optimizer.step()
      if config.scheduler != "ReduceLROnPlateau":
        scheduler.step()

      data_time_avg.update(data_time)
      iter_time_avg.update(iter_timer.toc(False))

      pred = get_prediction(data_loader.dataset, soutput.F, target)
      score = precision_at_one(pred, target)
      losses.update(batch_loss, target.size(0))
      scores.update(score, target.size(0))

      if curr_iter >= config.max_iter:
        is_training = False
        break

      if curr_iter % config.stat_freq == 0 or curr_iter == 1:
        # lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
        lrs = ', '.join(['{:.3e}'.format(optimizer.param_groups[0]['lr'])])
        debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
            epoch, curr_iter,
            len(data_loader) // config.iter_size, losses.avg, lrs)
        debug_str += "Score {:.3f}\tData time: {:.4f}, Total iter time: {:.4f}".format(
            scores.avg, data_time_avg.avg, iter_time_avg.avg)
        logging.info(debug_str)
        # Reset timers
        data_time_avg.reset()
        iter_time_avg.reset()
        # Write logs
        writer.add_scalar('training/loss', losses.avg, curr_iter)
        writer.add_scalar('training/precision_at_1', scores.avg, curr_iter)
        writer.add_scalar('training/learning_rate', optimizer.param_groups[0]['lr'], curr_iter)
        losses.reset()
        scores.reset()

      # Save current status, save before val to prevent occational mem overflow
      if curr_iter % config.save_freq == 0:
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
                   best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
                   best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
                   best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc, best_val_acc_iter=best_val_acc_iter)

      # Validation
      if curr_iter % config.val_freq == 0:
        val_loss, val_score, val_part_iou, val_shape_iou = \
          validate(model, val_data_loader, writer, curr_iter, config, weights)
        if val_part_iou > best_val_part_iou:
          best_val_part_iou = val_part_iou
          best_val_part_iou_iter = curr_iter
          checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
                     best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
                     best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
                     best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc,
                     best_val_acc_iter=best_val_acc_iter, postfix='best_part_iou')
        if val_shape_iou > best_val_shape_iou:
          best_val_shape_iou = val_shape_iou
          best_val_shape_iou_iter = curr_iter
          checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
                     best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
                     best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
                     best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc,
                     best_val_acc_iter=best_val_acc_iter, postfix='best_shape_iou')
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_loss_iter = curr_iter
          checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
                     best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
                     best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
                     best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc,
                     best_val_acc_iter=best_val_acc_iter, postfix='best_loss')
        if val_score > best_val_acc:
          best_val_acc = val_score
          best_val_acc_iter = curr_iter
          checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
                     best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
                     best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
                     best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc,
                     best_val_acc_iter=best_val_acc_iter, postfix='best_acc')

        logging.info("Current best Part IoU: {:.3f} at iter {}".format(best_val_part_iou, best_val_part_iou_iter))
        logging.info("Current best Shape IoU: {:.3f} at iter {}".format(best_val_shape_iou, best_val_shape_iou_iter))
        logging.info("Current best Loss: {:.3f} at iter {}".format(best_val_loss, best_val_loss_iter))
        logging.info("Current best Score: {:.3f} at iter {}".format(best_val_acc, best_val_acc_iter))

        # Recover back
        model.train()

      # End of iteration
      curr_iter += 1

    if config.scheduler == "ReduceLROnPlateau":
      try:
        scheduler.step(val_loss)
      except UnboundLocalError:
        pass
    epoch += 1

  # Explicit memory cleanup
  if hasattr(data_iter, 'cleanup'):
    data_iter.cleanup()

  # Save the final model
  val_loss, val_score, val_part_iou, val_shape_iou = \
    validate(model, val_data_loader, writer, curr_iter, config, weights)
  if val_part_iou > best_val_part_iou:
    best_val_part_iou = val_part_iou
    best_val_part_iou_iter = curr_iter
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
               best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
               best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
               best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc,
               best_val_acc_iter=best_val_acc_iter, postfix='best_part_iou')
  if val_shape_iou > best_val_shape_iou:
    best_val_shape_iou = val_shape_iou
    best_val_shape_iou_iter = curr_iter
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
               best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
               best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
               best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc,
               best_val_acc_iter=best_val_acc_iter, postfix='best_shape_iou')
  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_val_loss_iter = curr_iter
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
               best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
               best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
               best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc,
               best_val_acc_iter=best_val_acc_iter, postfix='best_loss')
  if val_score > best_val_acc:
    best_val_acc = val_score
    best_val_acc_iter = curr_iter
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_part_iou=best_val_part_iou,
               best_val_part_iou_iter=best_val_part_iou_iter, best_val_shape_iou=best_val_shape_iou,
               best_val_shape_iou_iter=best_val_shape_iou_iter, best_val_loss=best_val_loss,
               best_val_loss_iter=best_val_loss_iter, best_val_acc=best_val_acc,
               best_val_acc_iter=best_val_acc_iter, postfix='best_acc')

  logging.info("Current best Part IoU: {:.3f} at iter {}".format(best_val_part_iou, best_val_part_iou_iter))
  logging.info("Current best Shape IoU: {:.3f} at iter {}".format(best_val_shape_iou, best_val_shape_iou_iter))
  logging.info("Current best Loss: {:.3f} at iter {}".format(best_val_loss, best_val_loss_iter))
  logging.info("Current best Score: {:.3f} at iter {}".format(best_val_acc, best_val_acc_iter))
