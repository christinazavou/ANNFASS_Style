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
from runners.common import plot_grad_flow
from runners.common_dp2 import IM_AE_MYMODEL_COMMON
from utils.io_helper import setup_logging
from utils.matplotlib_utils import plot_matrix

from utils import *
from modelAE_GD import *


class IM_AE(IM_AE_MYMODEL_COMMON):

    def __init__(self, config):

        if os.path.exists("/media/graphicslab/BigData/zavou"):
            self.local = True
        else:
            self.local = False

        log_dir = os.path.dirname(config.checkpoint_dir)
        setup_logging(log_dir)
        self.log = logging.getLogger(self.__class__.__name__)

        print(f"initializing {self.__class__.__name__}...")
        self.log.debug(f"initializing {self.__class__.__name__}...")
        print(f"config:\n{config}")
        self.log.debug(f"config:\n{config}")

        self.real_size = 256
        self.mask_margin = 8

        self.g_dim = config.gen_dim
        self.d_dim = config.disc_dim
        self.z_dim = config.style_dim
        self.param_alpha = config.alpha
        self.param_beta = config.beta

        self.g_steps = config.g_steps
        self.d_steps = config.d_steps
        self.r_steps = config.r_steps

        self.input_size = config.input_size
        self.output_size = config.output_size

        self.asymmetry = config.asymmetry

        self.save_epoch = 2

        self.sampling_threshold = 0.4

        self.render_view_id = 0
        if self.asymmetry: self.render_view_id = 6  # render side view for motorbike
        self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir

        self.datapath = config.datapath

        self.device = get_torch_device(config, self.log)

        self.bce_loss = nn.BCELoss()


        # load data
        print("preprocessing - start")
        self.log.debug("preprocessing - start")


        self.dset = BasicDataset(self.data_dir, self.datapath, config, self.log, filename=config.filename)
        self.dataset_len = len(self.dset)
        self.upsample_rate = self.dset.upsample_rate

        if config.train:
            self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
            self.visualize_content_styles(config)
            cv2.imwrite(config.sample_dir + "/a_content_style_0.png", self.imgout_0)

        # build model
        self.discriminator = discriminator(self.d_dim, 1)  # todo: try with given styles also
        self.discriminator.to(self.device)

        sigmoid = False
        if config.recon_loss == "BCE":
            sigmoid = True
        self.generator = generator_halfsize_x8_allstyles(self.g_dim, self.z_dim, sigmoid)
        self.generator.to(self.device)

        if self.z_dim == 8:
            self.style_encoder = style_encoder_8()
        elif self.z_dim == 16:
            self.style_encoder = style_encoder_16(pool_method=config.pooling, kernel=config.kernel,
                                                  dilation=config.dilation)
        elif self.z_dim == 32:
            self.style_encoder = style_encoder_32(pool_method=config.pooling, kernel=config.kernel,
                                                  dilation=config.dilation)
        elif self.z_dim == 64:
            self.style_encoder = style_encoder_64()
        elif self.z_dim == 128:
            self.style_encoder = style_encoder_128()
        else:
            raise Exception(f"unknown z_dim {self.z_dim}")
        self.style_encoder.to(self.device)

        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr)
        self.optimizer_se = torch.optim.Adam(self.style_encoder.parameters(), lr=config.se_lr)

        # pytorch does not have a checkpoint manager
        # have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 20
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name = 'IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

    @property
    def model_dir(self):
        return "ae"

    def train(self, config):

        if not config.debug:
            self.visualise_init(config.sample_dir)

        iter_counter = self.load(config.gpu)
        if not iter_counter:
            iter_counter = 0

        train_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(self.checkpoint_dir), "train_log"))

        start_time = time.time()
        training_epoch = config.epoch

        batch_index_list = np.arange(self.dataset_len)

        epoch_size = self.dataset_len
        if config.debug:
            epoch_size = 10

        epoch = iter_counter // self.dataset_len
        while epoch < training_epoch:
            np.random.shuffle(batch_index_list)

            self.discriminator.train()
            self.generator.train()
            self.style_encoder.train()

            for idx in range(epoch_size):
                iter_counter += 1

                # ready a fake image
                content_idx = batch_index_list[idx]
                style_idx = random.randint(0, self.dataset_len-1)

                content_data_dict = self.dset.__getitem__(content_idx)
                mask_content = content_data_dict['mask']
                Dmask_content = content_data_dict['Dmask']
                input_content = content_data_dict['input']

                style_data_dict = self.dset.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']
                Dmask_style = style_data_dict['Dmask_style']

                mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

                z_tensor_g = self.style_encoder(voxel_style, is_training=False)
                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)
                voxel_fake = voxel_fake.detach()

                # D step
                for dstep in range(config.d_steps):  # notice that generator is not reused in each dstep as its not updated

                    self.discriminator.zero_grad()

                    D_out = self.discriminator(voxel_style, is_training=True)
                    loss_d_real = torch.sum((D_out - 1) ** 2 * Dmask_style) / torch.sum(Dmask_style)
                    loss_d_real.backward()

                    D_out = self.discriminator(voxel_fake, is_training=True)
                    loss_d_fake = torch.sum(D_out ** 2 * Dmask_fake) / torch.sum(Dmask_fake)
                    loss_d_fake.backward()

                    self.optimizer_d.step()

                # recon step
                # reconstruct style image
                all_styles = []
                r_steps = self.r_steps if iter_counter < 5000 else 1  # means after 2 epochs in chairs
                for rstep in range(r_steps):
                    style_idx_2 = np.random.randint(self.dataset_len)

                    style_data_dict_2 = self.dset.__getitem__(style_idx_2)
                    voxel_style_2 = style_data_dict_2['voxel_style']
                    mask_style_2 = style_data_dict_2['mask']
                    input_style_2 = style_data_dict_2['input']

                    voxel_style_2 = torch.from_numpy(voxel_style_2).to(self.device).unsqueeze(0).unsqueeze(0)
                    mask_style_2 = torch.from_numpy(mask_style_2).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_style_2 = torch.from_numpy(input_style_2).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    self.style_encoder.zero_grad()
                    self.generator.zero_grad()

                    z_tensor2_g = self.style_encoder(voxel_style_2, is_training=True)
                    voxel_fake = self.generator(input_style_2, z_tensor2_g, mask_style_2, is_training=True)
                    all_styles.append(z_tensor2_g)

                    if config.recon_loss == 'MSE':
                        loss_r = torch.mean((voxel_style_2 - voxel_fake) ** 2) * self.param_beta
                    else:
                        loss_r = self.bce_loss(voxel_fake, voxel_style_2)*self.param_beta

                    loss_r.backward()

                    # print("se,ge,d (rstep):", self.style_encoder.conv_1.weight.grad.sum(),
                    #       self.generator.conv_1.weight.grad.sum(), self.discriminator.conv_1.weight.grad.sum())
                    if iter_counter % config.log_iter == 0 and rstep == 0:
                        gen_grads = plot_grad_flow(self.generator.named_parameters())
                        se_grads = plot_grad_flow(self.style_encoder.named_parameters())
                        train_writer.add_figure('generator_r_step_grads', gen_grads, iter_counter)
                        train_writer.add_figure('style_encoder_r_step_grads', se_grads, iter_counter)
                    self.optimizer_g.step()
                    self.optimizer_se.step()

                all_styles = torch.cat(all_styles).view(-1, self.z_dim)

                # G step (+style encoder in this step)
                for gstep in range(self.g_steps):
                    self.style_encoder.zero_grad()
                    self.generator.zero_grad()

                    z_tensor_g = self.style_encoder(voxel_style, is_training=True)
                    voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=True)

                    D_out = self.discriminator(voxel_fake, is_training=False)

                    loss_g = torch.sum((D_out - 1) ** 2 * Dmask_fake) * self.param_alpha / torch.sum(Dmask_fake)
                    loss_g.backward()
                    self.optimizer_g.step()
                    self.optimizer_se.step()

                if iter_counter % config.log_iter == 0:
                    print(
                        "Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                            epoch, training_epoch, time.time() - start_time, loss_d_real.item(), loss_d_fake.item(),
                            loss_r.item(), loss_g.item()))
                    self.log.debug(
                        "Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                            epoch, training_epoch, time.time() - start_time, loss_d_real.item(), loss_d_fake.item(),
                            loss_r.item(), loss_g.item()))
                    train_writer.add_scalar('loss_d_real', loss_d_real.item(), iter_counter)
                    train_writer.add_scalar('loss_d_fake', loss_d_fake.item(), iter_counter)
                    train_writer.add_scalar('loss_r', loss_r.item(), iter_counter)
                    train_writer.add_scalar('loss_g', loss_g.item(), iter_counter)

                    heatmap_fig = plot_matrix(all_styles.detach().cpu().numpy(), show=False)
                    train_writer.add_figure('some_styles', heatmap_fig, iter_counter)

                    gen_grads = plot_grad_flow(self.generator.named_parameters())
                    se_grads = plot_grad_flow(self.style_encoder.named_parameters())
                    train_writer.add_figure('generator_g_step_grads', gen_grads, iter_counter)
                    train_writer.add_figure('style_encoder_g_step_grads', se_grads, iter_counter)

                    self.visualise(config.sample_dir, f"iter{iter_counter}")

                if iter_counter % config.save_iter == 0:
                    self.save(iter_counter)

            epoch += 1
            print(
                "Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                    epoch, training_epoch, time.time() - start_time, loss_d_real.item(), loss_d_fake.item(),
                    loss_r.item(), loss_g.item()))
            self.log.debug(
                "Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                    epoch, training_epoch, time.time() - start_time, loss_d_real.item(), loss_d_fake.item(),
                    loss_r.item(), loss_g.item()))
            train_writer.add_scalar('loss_d_real', loss_d_real.item(), iter_counter)
            train_writer.add_scalar('loss_d_fake', loss_d_fake.item(), iter_counter)
            train_writer.add_scalar('loss_r', loss_r.item(), iter_counter)
            train_writer.add_scalar('loss_g', loss_g.item(), iter_counter)

            heatmap_fig = plot_matrix(all_styles.detach().cpu().numpy(), show=False)
            train_writer.add_figure(f'some_styles', heatmap_fig, iter_counter)

            gen_grads = plot_grad_flow(self.generator.named_parameters())
            se_grads = plot_grad_flow(self.style_encoder.named_parameters())
            train_writer.add_figure('generator_g_step_grads', gen_grads, iter_counter)
            train_writer.add_figure('style_encoder_g_step_grads', se_grads, iter_counter)

            self.visualise(config.sample_dir, f"epoch{epoch}")
            self.save(iter_counter)

        # if finish, save
        self.save(iter_counter)
