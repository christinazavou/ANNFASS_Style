import logging
import numpy as np
import os
import torch
import torch.nn.parallel as parallel
import random
from tensorboardX import SummaryWriter

from lib.train_ae import validate_ae
from lib.utils import checkpoint, Timer, AverageMeter, get_ae_criterion
from lib.solvers import initialize_optimizer, initialize_scheduler

from MinkowskiEngine import SparseTensor


def train_ae(model, data_loader, val_data_loader, devices, config, transform_data_fn=None):
  # Set up the train flag for batch normalization
  model.train()
  # Set target device
  target_device = devices[0]

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

  criterion = get_ae_criterion(config)
  # Copy the loss layer
  criterions = parallel.replicate(criterion, devices)

  train_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "train"))
  val_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "val"))


  # Train the network
  logging.info('===> Start training')
  best_val_loss, best_val_loss_iter, curr_iter, epoch, is_training, scheduler = initialize_training(config, model,
                                                                                                    optimizer,
                                                                                                    scheduler)

  data_iter = data_loader.__iter__()
  torch.autograd.set_detect_anomaly(True)
  num_gpus = len(devices)
  num_batches = (len(data_loader) + num_gpus - 1) // num_gpus
  while is_training:

    for iteration in range(num_batches):
      optimizer.zero_grad()
      iter_timer.tic()
      batch_loss = 0

      all_targets, data_time, inputs, outputs = one_feed_forward_step(config, data_iter, devices, model, num_gpus)

      # Extract features from the sparse tensors to use a pytorch criterion
      in_features = [input.F for input in inputs]
      out_features = [output.F for output in outputs]

      # if random.random() > 0.9:
      #   print("train input output min max : {},{}, {},{}".format(
      #     inputs[0].F.min(), inputs[0].F.max(), outputs[0].F.min(), outputs[0].F.max()))

      parallel_losses = parallel.parallel_apply(criterions, tuple(zip(in_features, out_features)), devices=devices)
      loss = parallel.gather(parallel_losses, target_device, dim=0).mean()

      # Compute and accumulate gradient
      batch_loss += loss.item()
      loss.backward()
      torch.cuda.empty_cache()

      # Update number of steps
      optimizer.step()
      if config.scheduler != "ReduceLROnPlateau":
        scheduler.step()

      data_time_avg.update(data_time)
      iter_time_avg.update(iter_timer.toc(False))

      all_targets_size = 0  # basically the amount of points
      for sub_iter in range(num_gpus):
        all_targets_size += all_targets[sub_iter].size(0)
      losses.update(batch_loss, all_targets_size)

      if curr_iter >= config.max_iter:
        is_training = False
        break

      if curr_iter % config.stat_freq == 0 or curr_iter == 1:
        log_iteration(curr_iter, data_time_avg, epoch, iter_time_avg, losses, num_batches, optimizer, scores,
                      train_writer)

      # Save current status, save before val to prevent occational mem overflow
      if curr_iter % config.save_freq == 0:
        checkpoint(model=model, optimizer=optimizer, epoch=epoch, iteration=curr_iter, config=config,
                   postfix='iter_{}'.format(curr_iter))

      # Validation
      if curr_iter % config.val_freq == 0:
        best_val_loss, best_val_loss_iter, val_loss = run_validation_step(best_val_loss, best_val_loss_iter, config,
                                                                          curr_iter, epoch, model, optimizer,
                                                                          val_data_loader, val_writer, criterion)
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
  best_val_loss, best_val_loss_iter, val_loss = run_validation_step(best_val_loss, best_val_loss_iter, config,
                                                                    curr_iter, epoch, model, optimizer,
                                                                    val_data_loader, val_writer, criterion)
  logging.info("Final best Loss: {:.3f} at iter {}".format(best_val_loss, best_val_loss_iter))


def run_validation_step(best_val_loss, best_val_loss_iter, config, curr_iter, epoch, model, optimizer, val_data_loader,
                        val_writer, criterion):
  if len(val_data_loader) == 0:
    checkpoint(model=model, optimizer=optimizer, epoch=epoch, iteration=curr_iter, config=config, postfix='last_iter')
    return None, None, None
  val_loss = validate_ae(model, val_data_loader, val_writer, curr_iter, criterion, config)
  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_val_loss_iter = curr_iter
    checkpoint(model=model, optimizer=optimizer, epoch=epoch, iteration=curr_iter, config=config,
               best_val_loss=best_val_loss, best_val_loss_iter=best_val_loss_iter, postfix='best_loss')
  logging.info("Current best Loss: {:.3f} at iter {}".format(best_val_loss, best_val_loss_iter))
  return best_val_loss, best_val_loss_iter, val_loss


def log_iteration(curr_iter, data_time_avg, epoch, iter_time_avg, losses, num_batches, optimizer, scores,
                  train_writer):
  lrs = ', '.join(['{:.3e}'.format(optimizer.param_groups[0]['lr'])])
  debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
    epoch, curr_iter,
    num_batches, losses.avg, lrs)
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


def one_feed_forward_step(config, data_iter, devices, model, num_gpus):
  data_timer = Timer()
  data_time = 0
  # Get new data
  inputs, all_targets = [], []
  data_timer.tic()

  for sub_iter in range(num_gpus):
    # Get training data
    coords, input, target = data_iter.next()

    # Preprocess input
    if config.normalize_color:
      # For BuildNet
      input[:, :3] = input[:, :3] / 255. - config.color_offset

    with torch.cuda.device(devices[sub_iter]):
      inputs.append(SparseTensor(input, coords).to(devices[sub_iter]))

    all_targets.append(target.to(devices[sub_iter]))

  data_time += data_timer.toc(False)

  # Feed forward
  # The raw version of the parallel_apply
  replicas = parallel.replicate(model, devices)
  outputs = parallel.parallel_apply(replicas, inputs, devices=devices)
  return all_targets, data_time, inputs, outputs


def initialize_training(config, model, optimizer, scheduler):
  best_val_loss, best_val_loss_iter, curr_iter, epoch, is_training = \
    np.Inf, 0, 1, 1, True
  if config.resume:
    checkpoint_fn = config.resume
    if os.path.isfile(checkpoint_fn):
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
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))
  return best_val_loss, best_val_loss_iter, curr_iter, epoch, is_training, scheduler

