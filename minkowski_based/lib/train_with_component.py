import logging
import os
import os.path as osp
import numpy as np
import random
import torch

from tensorboardX import SummaryWriter

from lib.test_with_component import test, test_cls_style
from lib.transforms_extended import get_component_indices_matrix
from lib.utils import checkpoint, Timer, AverageMeter, get_torch_device, get_with_component_criterion, get_class_weights
from lib.solvers import initialize_optimizer, initialize_scheduler

from MinkowskiEngine import SparseTensor

from models.modules.non_trainable_layers import get_average_per_component_t


def validate(model, val_data_loader, val_writer, curr_iter, criterion, config):
  loss, score = test_cls_style(model, val_data_loader, criterion, config)
  val_writer.add_scalar('loss', loss, curr_iter)
  if score is not None:
    val_writer.add_scalar('precision_at_1', score, curr_iter)
  return loss, score


def train(model, train_data_loader, val_data_loader, config, transform_data_fn=None):
  device = get_torch_device(config.is_cuda)
  # Set up the train flag for batch normalization
  model.train()

  # Configuration
  data_timer, iter_timer = Timer(), Timer()
  data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
  losses, scores = AverageMeter(), AverageMeter()

  val_size = len(val_data_loader)
  if val_size == 0:
    assert config.scheduler != "ReduceLROnPlateau", "Can't use ReduceLROnPlateau without validation data. " \
                                                    "Please use StepLR and provide step_size,step_gamma"

  optimizer = initialize_optimizer(model.parameters(), config)
  scheduler = initialize_scheduler(optimizer, config)

  class_weights = get_class_weights(config, train_data_loader.dataset)
  criterion = get_with_component_criterion(config, train_data_loader.dataset, class_weights)

  train_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "train"))
  val_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "val"))

  # Train the network
  logging.info('===> Start training')
  best_val_loss, best_val_loss_iter, best_val_score, best_val_score_iter, \
    curr_iter, epoch, is_training, scheduler = initialize_training(config, model, optimizer, scheduler)

  data_iter = train_data_loader.__iter__()
  torch.autograd.set_detect_anomaly(True)
  while is_training:

    for iteration in range(len(train_data_loader)):
      optimizer.zero_grad()
      iter_timer.tic()
      batch_loss = 0

      # Get training data
      data_time, loss, target = one_feed_forward_step(config, criterion, train_data_loader, data_iter, device, model)

      # print("weights: {}".format(model.log_weights()))

      # Compute gradient
      batch_loss += loss.item()
      loss.backward()
      torch.cuda.empty_cache()

      # Update number of steps
      optimizer.step()
      if config.scheduler != "ReduceLROnPlateau":
        scheduler.step()

      data_time_avg.update(data_time)
      iter_time_avg.update(iter_timer.toc(False))

      losses.update(batch_loss, target.size(0))

      if curr_iter >= config.max_iter:
        is_training = False
        break

      if curr_iter % config.stat_freq == 0 or curr_iter == 1:
        log_iteration(curr_iter, train_data_loader, data_time_avg, epoch, iter_time_avg, losses, optimizer, scores,
                      train_writer)

      # Save current status, save before val to prevent occational mem overflow
      if curr_iter % config.save_freq == 0:
        checkpoint(model=model, optimizer=optimizer, epoch=epoch, iteration=curr_iter, config=config,
                   postfix='iter_{}'.format(curr_iter))

      # Validation
      if curr_iter % config.val_freq == 0:
        best_val_loss, best_val_loss_iter, best_val_score, best_val_score_iter, val_loss, val_score = \
          run_validation_step(best_val_loss, best_val_loss_iter, best_val_score, best_val_score_iter, config,
                              curr_iter, epoch, model, optimizer, val_data_loader, val_writer, criterion)

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
  best_val_loss, best_val_loss_iter, best_val_score, best_val_score_iter, val_loss, val_score = \
    run_validation_step(best_val_loss, best_val_loss_iter, best_val_score, best_val_score_iter, config,
                        curr_iter, epoch, model, optimizer, val_data_loader, val_writer, criterion)

  logging.info("Final best Loss: {:.3f} at iter {}".format(best_val_loss, best_val_loss_iter))
  logging.info("Final best Score: {:.3f} at iter {}".format(best_val_score, best_val_score_iter))


def one_feed_forward_step(config, criterion, data_loader, data_iter, device, model):
  data_timer = Timer()
  data_timer.tic()
  coords, input, target, component_ids, component_names = data_iter.next()

  # Preprocess input
  if config.normalize_color:
    # For BuildNet
    input[:, :3] = input[:, :3] / 255. - config.color_offset

  sinput = SparseTensor(input, coords).to(device)

  if not "RNV" in data_loader.dataset.__class__.__name__ and not 'ROV' in data_loader.dataset.__class__.__name__:
    cimat, cnames = get_component_indices_matrix(component_ids, component_names)
    c_i_mat_t = torch.tensor(cimat)
    c_coords_t = get_average_per_component_t(c_i_mat_t, coords)
    c_target_t = get_average_per_component_t(c_i_mat_t, torch.unsqueeze(target, 1), is_target=True)
    sc_indices = SparseTensor(c_i_mat_t, c_coords_t).to(device)
    sc_target = SparseTensor(c_target_t, c_coords_t).to(device)

    data_time = data_timer.toc(False)

    # Feed forward
    inputs = (sinput, sc_indices)
    soutput = model(*inputs)

    if torch.isnan(sinput.F).any() or torch.isnan(soutput.F).any():
      print("OH NO! sinput.F or soutput.F contains nan")

    loss = criterion(soutput.F, torch.squeeze(sc_target.F.long()))  # todo: is this fine? what happens in a label is 5.6? do i have such label?
    return data_time, loss, sc_target.F
  else:
    data_time = data_timer.toc(False)

    # Feed forward
    inputs = (sinput, )
    soutput = model(*inputs)

    if torch.isnan(sinput.F).any() or torch.isnan(soutput.F).any():
      print("OH NO! sinput.F or soutput.F contains nan")

    target = target.long().to(device)
    loss = criterion(soutput.F, target.long())
    return data_time, loss, target


def log_iteration(curr_iter, data_loader, data_time_avg, epoch, iter_time_avg, losses, optimizer, scores,
                  train_writer):
  # lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
  lrs = ', '.join(['{:.3e}'.format(optimizer.param_groups[0]['lr'])])
  debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
    epoch, curr_iter, len(data_loader), losses.avg, lrs)
  debug_str += "Score {:.3f}\tData time: {:.4f}, Total iter time: {:.4f}".format(
    scores.avg, data_time_avg.avg, iter_time_avg.avg)
  logging.info(debug_str)
  # Reset timers
  data_time_avg.reset()
  iter_time_avg.reset()
  # Write logs
  train_writer.add_scalar('loss', losses.avg, curr_iter)
  train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], curr_iter)
  losses.reset()


def run_validation_step(best_val_loss, best_val_loss_iter, best_val_acc, best_val_acc_iter,
                        config, curr_iter, epoch, model, optimizer, val_data_loader,
                        val_writer, criterion):
  if len(val_data_loader) == 0:
    checkpoint(model=model, optimizer=optimizer, epoch=epoch, iteration=curr_iter, config=config, postfix='last_iter')
    return None, None, None
  val_loss, val_score = validate(model, val_data_loader, val_writer, curr_iter, criterion, config)
  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_val_loss_iter = curr_iter
    checkpoint(model=model, optimizer=optimizer, epoch=epoch, iteration=curr_iter, config=config,
               best_val_loss=best_val_loss, best_val_loss_iter=best_val_loss_iter,
               best_val_acc=best_val_acc, best_val_acc_iter=best_val_acc_iter, postfix='best_loss')
  if val_score > best_val_acc:
    best_val_acc = val_score
    best_val_acc_iter = curr_iter
    checkpoint(model=model, optimizer=optimizer, epoch=epoch, iteration=curr_iter, config=config,
               best_val_loss=best_val_loss, best_val_loss_iter=best_val_loss_iter,
               best_val_acc=best_val_acc, best_val_acc_iter=best_val_acc_iter, postfix='best_acc')

  logging.info("Current best Loss: {:.3f} at iter {}".format(best_val_loss, best_val_loss_iter))
  logging.info("Current best Acc: {:.3f} at iter {}".format(best_val_acc, best_val_acc_iter))
  return best_val_loss, best_val_loss_iter, best_val_acc, best_val_acc_iter, val_loss, val_score


def initialize_training(config, model, optimizer, scheduler):
  best_val_loss, best_val_loss_iter, best_val_score, best_val_score_iter, curr_iter, epoch, is_training = \
    np.Inf, 0, -np.Inf, 0, 1, 1, True
  if config.resume:
    checkpoint_fn = config.resume
    if osp.isfile(checkpoint_fn):
      logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn)
      curr_iter = state['iteration'] + 1
      epoch = state['epoch']
      model.load_state_dict(state['state_dict'])
      if config.resume_optimizer:
        scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
        optimizer.load_state_dict(state['optimizer'])
      if 'best_val_loss' in state:
        best_val_loss = state['best_val_loss']
        best_val_loss_iter = state['best_val_loss_iter']
      else:
        raise ValueError("=> no best_val_loss checkpoint (ae) found at '{}'".format(checkpoint_fn))
      if 'best_val_score' in state:
        best_val_score = state['best_val_score']
        best_val_score_iter = state['best_val_score_iter']
      else:
        raise ValueError("=> no best_val_score checkpoint (ae) found at '{}'".format(checkpoint_fn))
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))
  return best_val_loss, best_val_loss_iter, best_val_score, best_val_score_iter, \
         curr_iter, epoch, is_training, scheduler
