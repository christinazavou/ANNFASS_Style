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
from runners.common_dp2 import IM_AE_ADAIN_COMMON
from utils.io_helper import setup_logging
from utils.matplotlib_utils import plot_matrix

from utils import *
from modelAE_GD import *


class IM_AE(IM_AE_ADAIN_COMMON):

    def __init__(self, config):

        raise NotImplementedError()

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
        self.param_alpha = config.alpha
        self.param_beta = config.beta
        self.adain_alpha = config.adain_alpha

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
        self.style_dir = config.style_dir

        self.datapath = config.datapath
        self.stylepath = config.stylepath

        self.device = get_torch_device(config, self.log)

        self.bce_loss = nn.BCELoss()

        # load data
        print("preprocessing - start")
        self.log.debug("preprocessing - start")

        self.style_set = FlexDataset(self.style_dir,
                                     self.stylepath,
                                     dotdict({'cache_dir': config.style_cache_dir,
                                              'input_size': config.input_size,
                                              'output_size': config.output_size,
                                              'asymmetry': config.asymmetry,
                                              'gpu': config.gpu}),
                                     self.log,
                                     filename=config.style_filename)
        self.styleset_len = len(self.style_set)
        self.upsample_rate = self.style_set.upsample_rate

        if (config.train and not config.debug) or config.visualize_styles:
            self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
            self.visualize_styles(config)
            cv2.imwrite(config.sample_dir + "/a_style_0.png", self.imgout_0)

        self.dset = FlexDataset(self.data_dir,
                                self.datapath,
                                dotdict({'cache_dir': config.data_cache_dir,
                                         'input_size': config.input_size,
                                         'output_size': config.output_size,
                                         'asymmetry': config.asymmetry,
                                         'gpu': config.gpu}),
                                self.log,
                                filename=config.data_filename)
        self.dataset_len = len(self.dset)

        if (config.train and not config.debug) or config.visualize_contents:
            self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
            self.visualize_contents(config)
            cv2.imwrite(config.sample_dir + "/a_content_0.png", self.imgout_0)

        # build model
        self.discriminator = discriminator(self.d_dim, self.styleset_len + 1)
        self.discriminator.to(self.device)

        sigmoid = False
        if config.recon_loss == "BCE":
            sigmoid = True

        if self.input_size == 64 and self.output_size == 256:
            raise NotImplementedError()
        elif self.input_size == 32 and self.output_size == 128:
            raise NotImplementedError()
        elif self.input_size == 32 and self.output_size == 256:
            raise NotImplementedError()
        elif self.input_size == 16 and self.output_size == 128:
            self.generator = generator_halfsize_x8_adain_share(self.g_dim, sigmoid, self.adain_alpha)
        self.generator.to(self.device)

        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr)

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

        raise Exception('deprecated')

        print("WARNING: THIS PROBABLY DOESNT MAKE SENSE...I.E. TO USE SAME WEIGHTS BUT DIFFERENT DILATION AND STRIDE")

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

            for idx in range(epoch_size):
                iter_counter += 1

                # random a z vector for D training
                z_vector_style_idx = np.random.randint(self.styleset_len)

                # ready a fake image
                dxb = batch_index_list[idx]
                content_data_dict = self.dset.__getitem__(dxb)
                mask_content = content_data_dict['mask']
                Dmask_content = content_data_dict['Dmask']
                input_content = content_data_dict['input']

                style_data_dict = self.style_set.__getitem__(z_vector_style_idx)
                voxel_style = style_data_dict['voxel_style']
                Dmask_style = style_data_dict['Dmask_style']

                mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

                voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=False)
                voxel_fake = voxel_fake.detach()

                #D step
                for dstep in range(config.d_steps):  # notice that generator is not reused in each dstep as its not updated

                    self.discriminator.zero_grad()

                    D_out = self.discriminator(voxel_style, is_training=True)
                    # loss of "real specific style" + loss of "real any style"
                    loss_d_real = (torch.sum(
                        (D_out[:, z_vector_style_idx:z_vector_style_idx + 1] - 1) ** 2 * Dmask_style) +
                                   torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_style)) / torch.sum(Dmask_style)
                    loss_d_real.backward()

                    D_out = self.discriminator(voxel_fake, is_training=True)
                    # loss of "fake specific style" + loss of "fake any style"
                    loss_d_fake = (torch.sum((D_out[:, z_vector_style_idx:z_vector_style_idx + 1]) ** 2 * Dmask_fake) +
                                   torch.sum((D_out[:, -1:]) ** 2 * Dmask_fake)) / torch.sum(Dmask_fake)
                    loss_d_fake.backward()

                    self.optimizer_d.step()
                    self.generator.set_shared_weights(self.discriminator)

                # recon step
                # reconstruct style image
                r_steps = self.r_steps if iter_counter < 5000 else 1  # means after 2 epochs in chairs
                for rstep in range(r_steps):
                    qxp = np.random.randint(self.styleset_len)

                    style_data_dict_2 = self.style_set.__getitem__(qxp)
                    voxel_style_2 = style_data_dict_2['voxel_style']
                    mask_style_2 = style_data_dict_2['mask']
                    input_style_2 = style_data_dict_2['input']

                    voxel_style_2 = torch.from_numpy(voxel_style_2).to(self.device).unsqueeze(0).unsqueeze(0)
                    mask_style_2 = torch.from_numpy(mask_style_2).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_style_2 = torch.from_numpy(input_style_2).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    self.generator.zero_grad()

                    voxel_fake = self.generator(input_style_2, voxel_style_2, mask_style_2, is_training=True)

                    if config.recon_loss == 'MSE':
                        loss_r = torch.mean((voxel_style_2 - voxel_fake) ** 2) * self.param_beta
                    else:
                        loss_r = self.bce_loss(voxel_fake, voxel_style_2) * self.param_beta
                    loss_r.backward()
                    self.optimizer_g.step()
                    self.discriminator.set_shared_weights(self.generator)

                # G step
                for gstep in range(self.g_steps):
                    self.generator.zero_grad()

                    voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=True)

                    D_out = self.discriminator(voxel_fake, is_training=False)

                    loss_g = (torch.sum((D_out[:,
                                         z_vector_style_idx:z_vector_style_idx + 1] - 1) ** 2 * Dmask_fake) * self.param_alpha + torch.sum(
                        (D_out[:, -1:] - 1) ** 2 * Dmask_fake)) / torch.sum(Dmask_fake)
                    loss_g.backward()
                    self.optimizer_g.step()
                    self.discriminator.set_shared_weights(self.generator)

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

                    # gen_grads = plot_grad_flow(self.generator.named_parameters())
                    # train_writer.add_figure('generator_g_step_grads', gen_grads, iter_counter)

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

            # gen_grads = plot_grad_flow(self.generator.named_parameters())
            # train_writer.add_figure('generator_g_step_grads', gen_grads, iter_counter)

            self.visualise(config.sample_dir, f"epoch{epoch}")
            self.save(iter_counter)

            if epoch % 1 == 0:
                cv2.imwrite(config.sample_dir + "/" + str(epoch) + "_0.png", self.imgout_0)

        # if finish, save
        self.save(iter_counter)
