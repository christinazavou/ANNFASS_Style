import os
import logging
import time
import math
import random
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from sklearn.manifold import TSNE

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataset2 import *
from runners.common import plot_grad_flow, IM_AE_STATIC
from runners.common_dp2 import IM_AE_ADAIN_COMMON
from utils.io_helper import setup_logging
from utils.matplotlib_utils import plot_matrix

from utils import *
from modelAE_GD import *


class IM_AE(IM_AE_ADAIN_COMMON):

    def __init__(self, config):

        self._init_common_config(config)

        self.adain_alpha = config.adain_alpha
        self.optim = config.optim

        self._init_data(config)

        self.valpath = config.valpath
        self.val_dir = config.val_dir
        self.vset = FlexDataset(self.val_dir,
                                self.valpath,
                                dotdict({'cache_dir': config.val_cache_dir,
                                         'input_size': config.input_size,
                                         'output_size': config.output_size,
                                         'asymmetry': config.asymmetry,
                                         'gpu': config.gpu}),
                                self.log,
                                filename=config.val_filename)
        self.valset_len = len(self.vset)

        if (config.train and not config.debug) or config.visualize_validation:
            self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
            self.visualize_validation(config)
            cv2.imwrite(config.sample_dir + "/a_val_0.png", self.imgout_0)

        # build model
        self.discriminator = discriminator(self.d_dim, self.styleset_len + 1, init_weights=config.init_weights)
        self.discriminator.to(self.device)

        sigmoid = False
        if config.recon_loss == "BCE":
            sigmoid = True

        if self.input_size == 64 and self.output_size == 256:
            self.generator = generator_adain(self.g_dim, sigmoid, self.adain_alpha, init_weights=config.init_weights)
        elif self.input_size == 32 and self.output_size == 128:
            self.generator = generator_halfsize_adain(self.g_dim, sigmoid, self.adain_alpha, init_weights=config.init_weights)
        elif self.input_size == 32 and self.output_size == 256:
            self.generator = generator_halfsize_x8_adain(self.g_dim, sigmoid, self.adain_alpha, init_weights=config.init_weights)
        elif self.input_size == 16 and self.output_size == 128:
            self.generator = generator_halfsize_x8_adain(self.g_dim, sigmoid, self.adain_alpha, init_weights=config.init_weights)
        self.generator.to(self.device)

        if self.optim == 'Adam':
            self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr)
            self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr)
        elif self.optim == 'SGD':
            self.optimizer_d = torch.optim.SGD(self.discriminator.parameters(), lr=config.lr)
            self.optimizer_g = torch.optim.SGD(self.generator.parameters(), lr=config.lr)
        else:
            raise Exception(f"unknown optim {self.optim}")

    @property
    def model_dir(self):
        return "ae"

    def discriminator_step(self, content_batch, style_batch):  # real and fake are detailed shapes
        len_batch = len(content_batch)
        self.discriminator.zero_grad()

        loss_d_real_value = 0.
        loss_d_fake_value = 0.
        for (coarse_content, mask_content, Dmask_content), \
                (detailed_style, Dmask_style, style_idx) in zip(content_batch, style_batch):

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(coarse_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_style = torch.from_numpy(detailed_style).to(self.device).unsqueeze(0).unsqueeze(0)
            Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=False)
            voxel_fake = voxel_fake.detach()

            D_out = self.discriminator(voxel_style, is_training=True)
            # loss of "real specific style" + loss of "real any style"
            loss_d_real = (torch.sum(
                (D_out[:, style_idx:style_idx + 1] - 1) ** 2 * Dmask_style) +
                           torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_style)) / torch.sum(Dmask_style)
            loss_d_real = loss_d_real / len_batch
            loss_d_real_value += loss_d_real.item()
            loss_d_real.backward()

            D_out = self.discriminator(voxel_fake, is_training=True)
            # loss of "fake specific style" + loss of "fake any style"
            loss_d_fake = (torch.sum((D_out[:, style_idx:style_idx + 1]) ** 2 * Dmask_fake) +
                           torch.sum((D_out[:, -1:]) ** 2 * Dmask_fake)) / torch.sum(Dmask_fake)
            loss_d_fake = loss_d_fake / len_batch
            loss_d_fake_value += loss_d_fake.item()
            loss_d_fake.backward()

        self.optimizer_d.step()

        return loss_d_real_value, loss_d_fake_value

    def reconstruction_step(self, real_style_batch):
        len_batch = len(real_style_batch)
        self.generator.zero_grad()

        loss_r_value = 0.
        for (real_style_coarse, real_style_detailed, mask) in real_style_batch:
            real_style_detailed = torch.from_numpy(real_style_detailed).to(self.device).unsqueeze(0).unsqueeze(0)
            mask = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            real_style_coarse = torch.from_numpy(real_style_coarse).to(self.device).unsqueeze(0).unsqueeze(0).float()

            fake = self.generator(real_style_coarse, real_style_detailed, mask, is_training=True)
            loss_r = torch.mean((real_style_detailed - fake) ** 2) * self.param_beta
            loss_r = loss_r / len_batch

            loss_r_value += loss_r.item()
            loss_r.backward()

        self.optimizer_g.step()
        return loss_r_value

    def generator_step(self, content_batch, style_batch):
        len_batch = len(content_batch)
        self.generator.zero_grad()
        loss_g_value = 0.
        for (coarse_content, mask_content, Dmask_content), \
                (detailed_style, Dmask_style, style_idx) in zip(content_batch, style_batch):

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(coarse_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_style = torch.from_numpy(detailed_style).to(self.device).unsqueeze(0).unsqueeze(0)

            voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=True)
            D_out = self.discriminator(voxel_fake, is_training=False)

            loss_g = (torch.sum((D_out[:, style_idx:style_idx + 1] - 1) ** 2 * Dmask_fake) * self.param_alpha
                      + torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_fake)) / torch.sum(Dmask_fake)
            loss_g = loss_g / len_batch
            loss_g_value += loss_g.item()
            loss_g.backward()

        self.optimizer_g.step()
        return loss_g_value

    def eval_step(self, content_batch, style_batch):
        len_batch = len(content_batch)

        loss_d_real_value = 0.
        loss_d_fake_value = 0.
        loss_g_value = 0.
        loss_r_value = 0.

        for (coarse_content, mask_content, Dmask_content), \
                (coarse_style, detailed_style, mask_style, Dmask_style, style_idx) in zip(content_batch, style_batch):

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(coarse_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_style = torch.from_numpy(detailed_style).to(self.device).unsqueeze(0).unsqueeze(0)
            Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=False)
            voxel_fake = voxel_fake.detach()

            D_out = self.discriminator(voxel_style, is_training=True)

            loss_d_real = (torch.sum(
                (D_out[:, style_idx:style_idx + 1] - 1) ** 2 * Dmask_style) +
                           torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_style)) / torch.sum(Dmask_style)
            loss_d_real = loss_d_real / len_batch
            loss_d_real_value += loss_d_real.item()

            D_out = self.discriminator(voxel_fake, is_training=True)
            # loss of "fake specific style" + loss of "fake any style"
            loss_d_fake = (torch.sum((D_out[:, style_idx:style_idx + 1]) ** 2 * Dmask_fake) +
                           torch.sum((D_out[:, -1:]) ** 2 * Dmask_fake)) / torch.sum(Dmask_fake)
            loss_d_fake = loss_d_fake / len_batch
            loss_d_fake_value += loss_d_fake.item()

            loss_g = (torch.sum((D_out[:, style_idx:style_idx + 1] - 1) ** 2 * Dmask_fake) * self.param_alpha
                      + torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_fake)) / torch.sum(Dmask_fake)
            loss_g = loss_g / len_batch
            loss_g_value += loss_g.item()

            mask_style = torch.from_numpy(mask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

            fake = self.generator(coarse_style, detailed_style, mask_style, is_training=True)

            loss_r = torch.mean((voxel_style - fake) ** 2) * self.param_beta
            loss_r = loss_r / len_batch
            loss_r_value += loss_r.item()

        return loss_d_real_value, loss_d_fake_value,\
               loss_g_value, \
               loss_r_value

    def train(self, config):

        if not config.debug:
            self.visualise_init(config.sample_dir)

        iter_counter = self.load(config.gpu)
        if not iter_counter:
            iter_counter = 0

        train_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(self.checkpoint_dir), "train_log"))
        val_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(self.checkpoint_dir), "val_log"))

        start_time = time.time()
        training_epoch = config.epoch

        batch_index_list = np.arange(self.dataset_len)

        epoch_size = self.dataset_len
        if config.debug:
            epoch_size = 10

        epoch = iter_counter // (self.dataset_len // config.batch_size)
        while epoch < training_epoch:
            np.random.shuffle(batch_index_list)

            self.discriminator.train()
            self.generator.train()

            for idx in range(0, epoch_size-config.batch_size, config.batch_size):
                iter_counter += 1

                content_batch = []
                style_batch = []
                style_batch2 = []
                for b_idx in range(config.batch_size):

                    style_idx = np.random.randint(self.styleset_len)

                    dxb = batch_index_list[idx+b_idx]
                    content_data_dict = self.dset.__getitem__(dxb)
                    mask_content = content_data_dict['mask']
                    Dmask_content = content_data_dict['Dmask']
                    input_content = content_data_dict['input']

                    style_data_dict = self.style_set.__getitem__(style_idx)
                    voxel_style = style_data_dict['voxel_style']
                    Dmask_style = style_data_dict['Dmask_style']
                    mask_style = style_data_dict['mask']
                    coarse_style = style_data_dict['input']

                    content_batch.append((input_content, mask_content, Dmask_content))
                    style_batch.append((voxel_style, Dmask_style, style_idx))
                    style_batch2.append((coarse_style, voxel_style, mask_style))

                loss_d_real_values = []
                loss_d_fake_values = []
                for step in range(self.d_steps):  # notice that generator is not reused in each dstep as its not updated
                    loss_d_real_value, loss_d_fake_value = self.discriminator_step(content_batch, style_batch)
                    loss_d_real_values.append(loss_d_real_value)
                    loss_d_fake_values.append(loss_d_fake_value)
                loss_d_real_value = np.mean(loss_d_real_values)
                loss_d_fake_value = np.mean(loss_d_fake_values)

                loss_r_values = []
                r_steps = self.r_steps if iter_counter < 5000 // config.batch_size else 1  # means after 2 epochs in chairs
                for step in range(r_steps):
                    loss_r_value = self.reconstruction_step(style_batch2, )
                    loss_r_values.append(loss_r_value)
                loss_r_value = np.mean(loss_r_values)

                loss_g_values = []
                for step in range(self.g_steps):
                    loss_g_value = self.generator_step(content_batch, style_batch)
                    loss_g_values.append(loss_g_value)
                loss_g_value = np.mean(loss_g_values)

                if iter_counter % config.log_iter == 0:
                    print("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                           epoch, training_epoch, time.time() - start_time, loss_d_real_value, loss_d_fake_value,
                           loss_r_value, loss_g_value))
                    self.log.debug("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                                    epoch, training_epoch, time.time() - start_time, loss_d_real_value, loss_d_fake_value,
                                    loss_r_value, loss_g_value))
                    train_writer.add_scalar('loss_d_real', loss_d_real_value, iter_counter)
                    train_writer.add_scalar('loss_d_fake', loss_d_fake_value, iter_counter)
                    train_writer.add_scalar('loss_r', loss_r_value, iter_counter)
                    train_writer.add_scalar('loss_g', loss_g_value, iter_counter)

                    with torch.no_grad():
                        self.eval_iter(config, iter_counter, epoch, training_epoch, val_writer)

                if iter_counter % config.save_iter == 0:
                    self.save(iter_counter)

                torch.cuda.empty_cache()
            epoch += 1
            with torch.no_grad():
                self.visualise(config.sample_dir, f"epoch{epoch}")

        # if finish, save
        self.save(iter_counter)

    def eval_iter(self, config, iter_counter, epoch, training_epoch, val_writer):

        start_time = time.time()

        val_size = self.valset_len
        if config.debug:
            val_size = 10

        loss_d_real = []
        loss_d_fake = []
        loss_g = []
        loss_r = []

        for idx in range(0, val_size - config.batch_size, config.batch_size):
            content_batch = []
            style_batch = []
            for b_idx in range(config.batch_size):
                dxb = idx + b_idx

                style_idx = np.random.randint(self.styleset_len)

                content_data_dict = self.vset.__getitem__(dxb)
                mask_content = content_data_dict['mask']
                Dmask_content = content_data_dict['Dmask']
                input_content = content_data_dict['input']

                style_data_dict = self.style_set.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']
                Dmask_style = style_data_dict['Dmask_style']
                coarse_style = style_data_dict['input']
                mask_style = style_data_dict['mask']

                content_batch.append((input_content, mask_content, Dmask_content))
                style_batch.append((coarse_style, voxel_style, mask_style, Dmask_style, style_idx))

            loss_d_real_value, loss_d_fake_value, loss_g_value, loss_r_value = self.eval_step(content_batch, style_batch)

            loss_d_real.append(loss_d_real_value)
            loss_d_fake.append(loss_d_fake_value)
            loss_g.append(loss_g_value)
            loss_r.append(loss_r_value)

        loss_d_real = np.mean(loss_d_real)
        loss_d_fake = np.mean(loss_d_fake)
        loss_g = np.mean(loss_g)
        loss_r = np.mean(loss_r)
        print("Val Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, "
                          "loss_r: %.6f, loss_g: %.6f, " % (
                        epoch, training_epoch, time.time() - start_time,
                        loss_d_real, loss_d_fake, loss_r, loss_g))
        self.log.debug("Val Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, "
                          "loss_r: %.6f, loss_g: %.6f," % (
                        epoch, training_epoch, time.time() - start_time,
                        loss_d_real, loss_d_fake, loss_r, loss_g))
        val_writer.add_scalar('loss_d_real', loss_d_real, iter_counter)
        val_writer.add_scalar('loss_d_fake', loss_d_fake, iter_counter)
        val_writer.add_scalar('loss_g', loss_g, iter_counter)
        val_writer.add_scalar(f'loss_r*{self.param_beta}', loss_r, iter_counter)
