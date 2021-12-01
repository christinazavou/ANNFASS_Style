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
from runners.common import IM_AE_STATIC, plot_grad_flow
from runners.common_dp2 import IM_AE_ORIG_COMMON
from utils.io_helper import setup_logging
from utils.open3d_render import render_geometries
from utils.open3d_utils import TriangleMesh
from utils.pytorch3d_vis import CustomDefinedViewMeshRenderer
from utils.matplotlib_utils import render_result, render_example, render_views, plot_matrix

from utils import *
from modelAE_GD import *
import mcubes
from PIL import Image


class IM_AE(IM_AE_ORIG_COMMON):

    def __init__(self, config):

        self._init_common_config(config)

        self.z_dim = config.style_dim
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

        if self.input_size == 64 and self.output_size == 256:
            self.generator = generator(self.g_dim, self.styleset_len, self.z_dim, init_weights=config.init_weights)
        elif self.input_size == 32 and self.output_size == 128:
            self.generator = generator_halfsize(self.g_dim, self.styleset_len, self.z_dim, init_weights=config.init_weights)
        elif self.input_size == 32 and self.output_size == 256:
            self.generator = generator_halfsize_x8(self.g_dim, self.styleset_len, self.z_dim, init_weights=config.init_weights)
        elif self.input_size == 16 and self.output_size == 128:
            self.generator = generator_halfsize_x8(self.g_dim, self.styleset_len, self.z_dim, init_weights=config.init_weights)
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

    def discriminator_step(self, content_batch, style_batch, store_grads=False):  # real and fake are detailed shapes
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

            z_vector = np.zeros([self.styleset_len], np.float32)
            z_vector[style_idx] = 1
            z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])
            z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1, -1, 1, 1, 1])
            voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=True)
            voxel_fake = voxel_fake.detach()

            D_out = self.discriminator(voxel_style, is_training=True)
            d_out_global = D_out[:, -1:]
            d_out_style = D_out[:, style_idx:style_idx + 1]

            loss_d_real_style = torch.sum(torch.log(d_out_style + 1e-8) * Dmask_style) / torch.sum(Dmask_style)
            loss_d_real_global = torch.sum(torch.log(d_out_global + 1e-8) * Dmask_style) / torch.sum(Dmask_style)
            loss_d_real = loss_d_real_style + loss_d_real_global
            loss_d_real = loss_d_real / len_batch
            loss_d_real_value += loss_d_real.item()
            loss_d_real.backward()

            D_out = self.discriminator(voxel_fake, is_training=True)
            d_out_global = D_out[:, -1:]
            d_out_style = D_out[:, style_idx:style_idx + 1]
            loss_d_fake_style = torch.sum(torch.log(1 - d_out_style + 1e-8) * Dmask_fake) / torch.sum(Dmask_fake)
            loss_d_fake_global = torch.sum(torch.log(1 - d_out_global + 1e-8) * Dmask_fake) / torch.sum(Dmask_fake)
            loss_d_fake = loss_d_fake_style + loss_d_fake_global
            loss_d_fake = loss_d_fake / len_batch
            loss_d_fake_value += loss_d_fake.item()
            loss_d_fake.backward()

        if store_grads:
            gen_grads = plot_grad_flow(self.generator.named_parameters())
            disc_grads = plot_grad_flow(self.discriminator.named_parameters())
        else:
            gen_grads = None
            disc_grads = None

        self.optimizer_d.step()

        return loss_d_real_value, loss_d_fake_value, gen_grads, disc_grads

    def reconstruction_step(self, real_style_batch, store_grads=False):
        len_batch = len(real_style_batch)
        self.generator.zero_grad()

        loss_r_value = 0.
        for (real_style_coarse, real_style_detailed, mask, style_idx) in real_style_batch:
            real_style_detailed = torch.from_numpy(real_style_detailed).to(self.device).unsqueeze(0).unsqueeze(0)
            mask = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            real_style_coarse = torch.from_numpy(real_style_coarse).to(self.device).unsqueeze(0).unsqueeze(0).float()

            z_vector = np.zeros([self.styleset_len], np.float32)
            z_vector[style_idx] = 1
            z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])
            z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1, -1, 1, 1, 1])
            fake = self.generator(real_style_coarse, z_tensor_g, mask, is_training=True)
            loss_r = torch.mean((real_style_detailed - fake) ** 2) * self.param_beta
            loss_r /= len_batch

            loss_r_value += loss_r.item()
            loss_r.backward()

        if store_grads:
            gen_grads = plot_grad_flow(self.generator.named_parameters())
        else:
            gen_grads = None

        self.optimizer_g.step()
        return loss_r_value, gen_grads

    def generator_step(self, content_batch, style_batch, store_grads=False):
        len_batch = len(content_batch)
        self.generator.zero_grad()
        loss_g_value = 0.
        for (coarse_content, mask_content, Dmask_content), \
                (detailed_style, Dmask_style, style_idx) in zip(content_batch, style_batch):

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(coarse_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            z_vector = np.zeros([self.styleset_len], np.float32)
            z_vector[style_idx] = 1
            z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])
            z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1, -1, 1, 1, 1])
            voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=True)
            D_out = self.discriminator(voxel_fake, is_training=False)
            d_out_global = D_out[:, -1:]
            d_out_style = D_out[:, style_idx:style_idx + 1]

            loss_g_global = torch.sum(torch.log(1 - d_out_style + 1e-8) * Dmask_fake) * self.param_alpha / torch.sum(Dmask_fake)
            loss_g_style = torch.sum(torch.log(1 - d_out_global + 1e-8) * Dmask_fake) / torch.sum(Dmask_fake)
            loss_g = loss_g_style + loss_g_global
            loss_g = loss_g / len_batch
            loss_g_value += loss_g.item()
            loss_g.backward()

        if store_grads:
            gen_grads = plot_grad_flow(self.generator.named_parameters())
        else:
            gen_grads = None

        self.optimizer_g.step()
        return loss_g_value, gen_grads

    def disc_gen_eval_step(self, content_batch, style_batch):  # real and fake are detailed shapes
        len_batch = len(content_batch)

        loss_d_real_value = 0.
        loss_d_fake_value = 0.
        loss_g_value = 0.

        style_batch_vectors = []

        for (coarse_content, mask_content, Dmask_content), \
            (detailed_style, Dmask_style, style_idx) in zip(content_batch, style_batch):
            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(coarse_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_style = torch.from_numpy(detailed_style).to(self.device).unsqueeze(0).unsqueeze(0)
            Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

            z_vector = np.zeros([self.styleset_len], np.float32)
            z_vector[style_idx] = 1
            z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])
            z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1, -1, 1, 1, 1])
            voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=True)
            voxel_fake = voxel_fake.detach()

            D_out = self.discriminator(voxel_style, is_training=True)
            d_out_global = D_out[:, -1:]
            d_out_style = D_out[:, style_idx:style_idx + 1]
            loss_d_real_style = torch.sum(torch.log(d_out_style + 1e-8) * Dmask_style) / torch.sum(Dmask_style)
            loss_d_real_global = torch.sum(torch.log(d_out_global + 1e-8) * Dmask_style) / torch.sum(Dmask_style)
            loss_d_real = loss_d_real_style + loss_d_real_global
            loss_d_real = loss_d_real / len_batch
            loss_d_real_value += loss_d_real.item()

            D_out = self.discriminator(voxel_fake, is_training=True)
            d_out_global = D_out[:, -1:]
            d_out_style = D_out[:, style_idx:style_idx + 1]
            loss_d_fake_style = torch.sum(torch.log(1 - d_out_style + 1e-8) * Dmask_fake) / torch.sum(Dmask_fake)
            loss_d_fake_global = torch.sum(torch.log(1 - d_out_global + 1e-8) * Dmask_fake) / torch.sum(Dmask_fake)
            loss_d_fake = loss_d_fake_style + loss_d_fake_global
            loss_d_fake = loss_d_fake / len_batch
            loss_d_fake_value += loss_d_fake.item()

            loss_g_global = torch.sum(torch.log(1 - d_out_style + 1e-8) * Dmask_fake) * self.param_alpha / torch.sum(Dmask_fake)
            loss_g_style = torch.sum(torch.log(1 - d_out_global + 1e-8) * Dmask_fake) / torch.sum(Dmask_fake)
            loss_g = loss_g_style + loss_g_global
            loss_g = loss_g / len_batch
            loss_g_value += loss_g.item()

            style_batch_vectors.append(z_tensor_g[:,:,0,0,0].detach().cpu().numpy())

        style_batch_vectors = np.vstack(style_batch_vectors)

        return loss_d_real_value, loss_d_fake_value, loss_g_value, style_batch_vectors

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

        # Let G train for a few steps before beginning to jointly train G
        # and D because MM GANs have trouble learning very early on in training
        np.random.shuffle(batch_index_list)
        for idx in range(25):

            self.discriminator.eval()
            self.generator.train()

            iter_counter += 1

            content_batch = []
            style_batch = []
            style_idx = np.random.randint(self.styleset_len)

            dxb = batch_index_list[idx]
            content_data_dict = self.dset.__getitem__(dxb)
            mask_content = content_data_dict['mask']
            Dmask_content = content_data_dict['Dmask']
            input_content = content_data_dict['input']

            style_data_dict = self.style_set.__getitem__(style_idx)
            voxel_style = style_data_dict['voxel_style']
            Dmask_style = style_data_dict['Dmask_style']

            content_batch.append((input_content, mask_content, Dmask_content))
            style_batch.append((voxel_style, Dmask_style, style_idx))

            loss_g_value, gen_grads = self.generator_step(content_batch, style_batch, False)

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
                    style_batch2.append((coarse_style, voxel_style, mask_style, style_idx))

                loss_d_real_values = []
                loss_d_fake_values = []
                for step in range(self.d_steps):  # notice that generator is not reused in each dstep as its not updated
                    loss_d_real_value, loss_d_fake_value, gen_grads, disc_grads = self.discriminator_step(
                        content_batch, style_batch, iter_counter%config.log_iter==0 and step==0)
                    loss_d_real_values.append(loss_d_real_value)
                    loss_d_fake_values.append(loss_d_fake_value)
                    if gen_grads:
                        train_writer.add_figure('generator_d_step_grads', gen_grads, iter_counter)
                        train_writer.add_figure('discriminator_d_step_grads', disc_grads, iter_counter)
                loss_d_real_value = np.mean(loss_d_real_values)
                loss_d_fake_value = np.mean(loss_d_fake_values)

                loss_r_values = []
                r_steps = self.r_steps if iter_counter < 5000 // config.batch_size else 1  # means after 2 epochs in chairs
                for step in range(r_steps):
                    loss_r_value, gen_grads = self.reconstruction_step(style_batch2,
                                                                       iter_counter%config.log_iter==0 and step==0)
                    loss_r_values.append(loss_r_value)
                    if gen_grads:
                        train_writer.add_figure('generator_r_step_grads', gen_grads, iter_counter)
                loss_r_value = np.mean(loss_r_values)

                loss_g_values = []
                for step in range(self.g_steps):
                    loss_g_value, gen_grads = self.generator_step(content_batch,
                                                                  style_batch,
                                                                  iter_counter%config.log_iter==0 and step==0)
                    loss_g_values.append(loss_g_value)
                    if gen_grads:
                        train_writer.add_figure('generator_g_step_grads', gen_grads, iter_counter)
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
                        self.visualise(config.sample_dir, f"iter{iter_counter}")

                if iter_counter % config.save_iter == 0:
                    self.save(iter_counter)

            epoch += 1
            self.eval_epoch(config, iter_counter, epoch, training_epoch, val_writer)

        # if finish, save
        self.save(iter_counter)

    def eval_epoch(self, config, iter_counter, epoch, training_epoch, val_writer):

        start_time = time.time()

        val_size = self.valset_len
        if config.debug:
            val_size = 10

        loss_d_real_value_epoch = []
        loss_d_fake_value_epoch = []
        loss_g_value_epoch = []

        with torch.no_grad():
            for idx in range(0, val_size - config.batch_size, config.batch_size):
                content_batch = []
                style_batch = []
                for b_idx in range(config.batch_size):
                    style_idx = np.random.randint(self.styleset_len)

                    dxb = idx + b_idx
                    content_data_dict = self.vset.__getitem__(dxb)
                    mask_content = content_data_dict['mask']
                    Dmask_content = content_data_dict['Dmask']
                    input_content = content_data_dict['input']

                    style_data_dict = self.style_set.__getitem__(style_idx)
                    voxel_style = style_data_dict['voxel_style']
                    Dmask_style = style_data_dict['Dmask_style']

                    content_batch.append((input_content, mask_content, Dmask_content))
                    style_batch.append((voxel_style, Dmask_style, style_idx))

                loss_d_real_value, loss_d_fake_value, loss_g_value, style_batch_vectors = self.disc_gen_eval_step(content_batch, style_batch)

                loss_d_real_value_epoch.append(loss_d_real_value)
                loss_d_fake_value_epoch.append(loss_d_fake_value)
                loss_g_value_epoch.append(loss_g_value)

            loss_d_real_value_epoch = np.mean(loss_d_real_value_epoch)
            loss_d_fake_value_epoch = np.mean(loss_d_fake_value_epoch)
            loss_g_value_epoch = np.mean(loss_g_value_epoch)
            print("Val Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_g: %.6f" % (
                epoch, training_epoch, time.time() - start_time, loss_d_real_value_epoch, loss_d_fake_value_epoch,
                loss_g_value_epoch))
            self.log.debug("Val Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_g: %.6f" % (
                epoch, training_epoch, time.time() - start_time, loss_d_real_value_epoch, loss_d_fake_value_epoch,
                loss_g_value_epoch))
            val_writer.add_scalar('loss_d_real', loss_d_real_value_epoch, iter_counter)
            val_writer.add_scalar('loss_d_fake', loss_d_fake_value_epoch, iter_counter)
            val_writer.add_scalar('loss_g', loss_g_value_epoch, iter_counter)
            heatmap_fig = plot_matrix(style_batch_vectors, show=False)
            val_writer.add_figure('some_styles', heatmap_fig, iter_counter)
