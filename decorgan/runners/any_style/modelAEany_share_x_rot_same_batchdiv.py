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

from runners.common import IM_AE_STATIC, plot_grad_flow
from runners.common_dp2 import IM_AE_ANY_SHARE_COMMON_ROT_SAME, IM_AE_TWO_DATASETS
from utils.matplotlib_utils import plot_matrix

from utils import *
from modelAE_GD import *
from utils.patch_generator import get_random_non_cube_patches, get_random_non_cube_triplet_patches, \
    get_random_non_cube_pair_patches


class IM_AE(IM_AE_ANY_SHARE_COMMON_ROT_SAME):

    def __init__(self, config):

        self._init_common_config(config)

        self.z_dim = config.style_dim

        self.stride_factor = config.stride_factor
        self.num_triplets = config.num_triplets
        self.num_pairs = config.num_pairs

        self.param_gamma = config.gamma
        self.param_delta = config.delta

        self.clamp_num = config.clamp_num

        if (config.test_fig_3 is True or config.export is True) and config.train is False:
            IM_AE_TWO_DATASETS._init_data(self, config)
        else:
            self._init_data(config)

        # build model
        if config.any_share_type == 1:
            self.log.debug("config.any_share_type 1")
            print("config.any_share_type 1")
            if config.group_norm:
                self.common_discriminator = common_discriminator_1_gn(self.d_dim, init_weights=config.init_weights)
                self.discriminator_global = discriminator_part_global_plausibility_1_gn(self.d_dim,
                                                                                        wasserstein=self.use_wc,
                                                                                        init_weights=config.init_weights)
                self.discriminator_style = discriminator_part_style_plausibility_1_gn(self.d_dim, self.z_dim,
                                                                                      init_weights=config.init_weights)
            else:
                self.common_discriminator = common_discriminator_1(self.d_dim, init_weights=config.init_weights)
                self.discriminator_global = discriminator_part_global_plausibility_1(self.d_dim,
                                                                                     wasserstein=self.use_wc,
                                                                                     init_weights=config.init_weights)
                self.discriminator_style = discriminator_part_style_plausibility_1(self.d_dim, self.z_dim, init_weights=config.init_weights)
        elif config.any_share_type == 2:
            self.log.debug("config.any_share_type 2")
            print("config.any_share_type 2")
            if config.group_norm:
                self.common_discriminator = common_discriminator_2_gn(self.d_dim, init_weights=config.init_weights)
                self.discriminator_global = discriminator_part_global_plausibility_2(self.d_dim,
                                                                                     wasserstein=self.use_wc,
                                                                                     init_weights=config.init_weights)
                self.discriminator_style = discriminator_part_style_plausibility_2_gn(self.d_dim, self.z_dim, init_weights=config.init_weights)
            else:
                self.common_discriminator = common_discriminator_2(self.d_dim, init_weights=config.init_weights)
                self.discriminator_global = discriminator_part_global_plausibility_2(self.d_dim,
                                                                                     wasserstein=self.use_wc,
                                                                                     init_weights=config.init_weights)
                self.discriminator_style = discriminator_part_style_plausibility_2(self.d_dim, self.z_dim, init_weights=config.init_weights)
        elif config.any_share_type == 3:
            self.log.debug("config.any_share_type 3")
            print("config.any_share_type 3")
            if config.group_norm:
                self.common_discriminator = common_discriminator_3_gn(self.d_dim, init_weights=config.init_weights)
                self.discriminator_global = discriminator_part_global_plausibility_3_gn(self.d_dim,
                                                                                        wasserstein=self.use_wc,
                                                                                        init_weights=config.init_weights)
                self.discriminator_style = discriminator_part_style_plausibility_3_gn(self.d_dim, self.z_dim, init_weights=config.init_weights)
            else:
                self.common_discriminator = common_discriminator_3(self.d_dim, init_weights=config.init_weights)
                self.discriminator_global = discriminator_part_global_plausibility_3(self.d_dim,
                                                                                     wasserstein=self.use_wc,
                                                                                     init_weights=config.init_weights)
                self.discriminator_style = discriminator_part_style_plausibility_3(self.d_dim, self.z_dim, init_weights=config.init_weights)
        else:
            self.log.debug("config.any_share_type 4")
            print("config.any_share_type 4")
            if config.group_norm:
                self.common_discriminator = common_discriminator_4_gn(self.d_dim, init_weights=config.init_weights)
                self.discriminator_global = discriminator_part_global_plausibility_4_gn(self.d_dim,
                                                                                        wasserstein=self.use_wc,
                                                                                        init_weights=config.init_weights)
                self.discriminator_style = discriminator_part_style_plausibility_4_gn(self.d_dim, self.z_dim, init_weights=config.init_weights)
            else:
                self.common_discriminator = common_discriminator_4(self.d_dim, init_weights=config.init_weights)
                self.discriminator_global = discriminator_part_global_plausibility_4(self.d_dim,
                                                                                     wasserstein=self.use_wc,
                                                                                     init_weights=config.init_weights)
                self.discriminator_style = discriminator_part_style_plausibility_4(self.d_dim, self.z_dim, init_weights=config.init_weights)

        self.common_discriminator.to(self.device)
        self.discriminator_global.to(self.device)
        self.discriminator_style.to(self.device)

        sigmoid = False

        if config.group_norm:
            if self.input_size == 64 and self.output_size == 256:
                self.generator = generator_allstyles_gn(self.g_dim, self.z_dim, sigmoid, init_weights=config.init_weights)
            elif self.input_size == 32 and self.output_size == 128:
                self.generator = generator_halfsize_allstyles_gn(self.g_dim, self.z_dim, sigmoid, init_weights=config.init_weights)
            elif self.input_size == 32 and self.output_size == 256:
                self.generator = generator_halfsize_x8_allstyles_gn(self.g_dim, self.z_dim, sigmoid, init_weights=config.init_weights)
            elif self.input_size == 16 and self.output_size == 128:
                self.generator = generator_halfsize_x8_allstyles_gn(self.g_dim, self.z_dim, sigmoid, init_weights=config.init_weights)
        else:
            if self.input_size == 64 and self.output_size == 256:
                self.generator = generator_allstyles(self.g_dim, self.z_dim, sigmoid, init_weights=config.init_weights)
            elif self.input_size == 32 and self.output_size == 128:
                self.generator = generator_halfsize_allstyles(self.g_dim, self.z_dim, sigmoid, init_weights=config.init_weights)
            elif self.input_size == 32 and self.output_size == 256:
                self.generator = generator_halfsize_x8_allstyles(self.g_dim, self.z_dim, sigmoid, init_weights=config.init_weights)
            elif self.input_size == 16 and self.output_size == 128:
                self.generator = generator_halfsize_x8_allstyles(self.g_dim, self.z_dim, sigmoid, init_weights=config.init_weights)
        self.generator.to(self.device)

        # NOTE: here, i will have a "global" plausibility discriminator and a
        # style discriminator that will share all their layers except the last one.
        # this approach is very similar to original approach !!

        if self.use_wc:
            self.optimizer_d_common = torch.optim.RMSprop(self.common_discriminator.parameters(), lr=config.lr,
                                                          weight_decay=config.weight_decay)
            self.optimizer_d_global = torch.optim.RMSprop(self.discriminator_global.parameters(), lr=config.lr,
                                                          weight_decay=config.weight_decay)
            self.optimizer_d_style = torch.optim.RMSprop(self.discriminator_style.parameters(), lr=config.lr,
                                                         weight_decay=config.weight_decay)
            self.optimizer_g = torch.optim.RMSprop(self.generator.parameters(), lr=config.lr,
                                                   weight_decay=config.weight_decay)
        elif config.optim == 'Adam':
            self.optimizer_d_common = torch.optim.Adam(self.common_discriminator.parameters(), lr=config.lr,
                                                       weight_decay=config.weight_decay)
            self.optimizer_d_global = torch.optim.Adam(self.discriminator_global.parameters(), lr=config.lr,
                                                       weight_decay=config.weight_decay)
            self.optimizer_d_style = torch.optim.Adam(self.discriminator_style.parameters(), lr=config.lr,
                                                      weight_decay=config.weight_decay)
            self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr,
                                                weight_decay=config.weight_decay)
        elif config.optim == 'SGD':
            self.optimizer_d_common = torch.optim.SGD(self.common_discriminator.parameters(), lr=config.lr,
                                                      weight_decay=config.weight_decay, momentum=0.9, dampening=0.1)
            self.optimizer_d_global = torch.optim.SGD(self.discriminator_global.parameters(), lr=config.lr,
                                                      weight_decay=config.weight_decay, momentum=0.9, dampening=0.1)
            self.optimizer_d_style = torch.optim.SGD(self.discriminator_style.parameters(), lr=config.lr,
                                                     weight_decay=config.weight_decay, momentum=0.9, dampening=0.1)
            self.optimizer_g = torch.optim.SGD(self.generator.parameters(), lr=config.lr,
                                               weight_decay=config.weight_decay, momentum=0.9, dampening=0.1)
        else:
            raise Exception(f"unknown optim {config.optim}")

        self.triplet_criterion = torch.nn.TripletMarginLoss(margin=config.margin, p=2, reduce=True, reduction='mean')

    @property
    def model_dir(self):
        return "ae"

    def discriminator_step(self, content_batch, style_batch, writer=None, iter_counter=None):  # real and fake are detailed shapes
        len_batch = len(content_batch)
        self.common_discriminator.zero_grad()
        self.discriminator_global.zero_grad()
        self.discriminator_style.zero_grad()

        if self.use_wc:
            for param in self.common_discriminator.parameters():
                param.data.clamp_(-self.clamp_num, self.clamp_num)
            for param in self.discriminator_global.parameters():
                param.data.clamp_(-self.clamp_num, self.clamp_num)

        loss_d_real_value = 0.
        loss_d_fake_value = 0.
        patch_loss_d_real_value = 0.
        patch_loss_d_fake_value = 0.
        for (coarse_content, mask_content, Dmask_content), \
                (detailed_style, Dmask_style, detailed_other_style) in zip(content_batch, style_batch):

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(coarse_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_style = torch.from_numpy(detailed_style).to(self.device).unsqueeze(0).unsqueeze(0)
            Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_other_style = torch.from_numpy(detailed_other_style).to(self.device).unsqueeze(0).unsqueeze(0)

            d_common_real = self.common_discriminator(voxel_style, is_training=False)
            z_tensor_g = self.discriminator_style(d_common_real, is_training=False)
            z_tensor_g = self.discriminator_style.pool(z_tensor_g)
            voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)
            voxel_fake = voxel_fake.detach()

            d_common_real = self.common_discriminator(voxel_style, is_training=True)
            d_global_real = self.discriminator_global(d_common_real, is_training=True)
            if self.use_wc:
                loss_d_real = - torch.sum(d_global_real * Dmask_style) / torch.sum(Dmask_style)
            else:
                loss_d_real = torch.sum((d_global_real - 1) ** 2 * Dmask_style) / torch.sum(Dmask_style)
            loss_d_real = loss_d_real / len_batch
            loss_d_real_value += loss_d_real.item()
            loss_d_real.backward(retain_graph=True)

            d_common_fake = self.common_discriminator(voxel_fake, is_training=True)
            d_global_fake = self.discriminator_global(d_common_fake, is_training=True)
            if self.use_wc:
                loss_d_fake = torch.sum(d_global_fake * Dmask_fake) / torch.sum(Dmask_fake)
            else:
                loss_d_fake = torch.sum(d_global_fake ** 2 * Dmask_fake) / torch.sum(Dmask_fake)
            loss_d_fake = loss_d_fake / len_batch
            loss_d_fake_value += loss_d_fake.item()
            loss_d_fake.backward(retain_graph=True)

            d_style_real = self.discriminator_style(d_common_real, is_training=True)
            d_style_fake = self.discriminator_style(d_common_fake, is_training=True)
            d_common_other = self.common_discriminator(voxel_other_style, is_training=True)
            d_style_other = self.discriminator_style(d_common_other, is_training=True)

            anchors, positives, negatives = get_random_non_cube_triplet_patches(d_style_real,
                                                                                d_style_other,
                                                                                stride_factor=self.stride_factor,
                                                                                num_triplets=self.num_triplets)
            anchors = self.discriminator_style.pool(anchors).view(-1, self.z_dim)
            positives = self.discriminator_style.pool(positives).view(-1, self.z_dim)
            negatives = self.discriminator_style.pool(negatives).view(-1, self.z_dim)

            patch_loss_d_real = self.triplet_criterion(anchors, positives, negatives) * self.param_gamma
            patch_loss_d_real = patch_loss_d_real / len_batch
            patch_loss_d_real_value += patch_loss_d_real.item()
            patch_loss_d_real.backward(retain_graph=True)

            anchors, positives, negatives = get_random_non_cube_triplet_patches(d_style_real,
                                                                                d_style_fake,
                                                                                stride_factor=self.stride_factor,
                                                                                num_triplets=self.num_triplets)
            anchors = self.discriminator_style.pool(anchors).view(-1, self.z_dim)
            positives = self.discriminator_style.pool(positives).view(-1, self.z_dim)
            negatives = self.discriminator_style.pool(negatives).view(-1, self.z_dim)

            patch_loss_d_fake = self.triplet_criterion(anchors, positives, negatives) * self.param_delta
            patch_loss_d_fake = patch_loss_d_fake / len_batch
            patch_loss_d_fake_value += patch_loss_d_fake.item()
            patch_loss_d_fake.backward(retain_graph=False)

        if writer:
            common_disc_grads = plot_grad_flow(self.common_discriminator.named_parameters())
            disc_global_grads = plot_grad_flow(self.discriminator_global.named_parameters())
            disc_style_grads = plot_grad_flow(self.discriminator_style.named_parameters())
            writer.add_figure('common_disc_grads', common_disc_grads, iter_counter)
            writer.add_figure('disc_global_grads', disc_global_grads, iter_counter)
            writer.add_figure('disc_style_grads', disc_style_grads, iter_counter)

        self.optimizer_d_common.step()
        self.optimizer_d_global.step()
        self.optimizer_d_style.step()

        return loss_d_real_value, loss_d_fake_value, patch_loss_d_real_value, patch_loss_d_fake_value

    def reconstruction_step(self, real_style_batch, writer=None, iter_counter=None):
        len_batch = len(real_style_batch)

        self.common_discriminator.zero_grad()
        self.discriminator_style.zero_grad()
        self.generator.zero_grad()

        loss_r_value = 0.
        for (real_style_coarse, real_style_detailed, mask) in real_style_batch:

            real_style_detailed = torch.from_numpy(real_style_detailed).to(self.device).unsqueeze(0).unsqueeze(0)
            mask = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            real_style_coarse = torch.from_numpy(real_style_coarse).to(self.device).unsqueeze(0).unsqueeze(0).float()

            d_common_real_2 = self.common_discriminator(real_style_detailed, is_training=True)
            z_tensor_g = self.discriminator_style(d_common_real_2, is_training=True)
            z_tensor_g = self.discriminator_style.pool(z_tensor_g)
            fake = self.generator(real_style_coarse, z_tensor_g, mask, is_training=True)

            loss_r = torch.mean((real_style_detailed - fake) ** 2) * self.param_beta
            loss_r = loss_r / len_batch

            loss_r_value += loss_r.item()
            loss_r.backward(retain_graph=False)

        if writer:
            common_disc_grads = plot_grad_flow(self.common_discriminator.named_parameters())
            disc_style_grads = plot_grad_flow(self.discriminator_style.named_parameters())
            writer.add_figure('common_disc_grads', common_disc_grads, iter_counter)
            writer.add_figure('disc_style_grads', disc_style_grads, iter_counter)

        self.optimizer_g.step()
        self.optimizer_d_common.step()
        self.optimizer_d_style.step()
        return loss_r_value

    def generator_step(self, content_batch, style_batch, writer=None, iter_counter=None):
        len_batch = len(content_batch)

        self.generator.zero_grad()
        self.common_discriminator.zero_grad()
        self.discriminator_style.zero_grad()

        loss_g_value = 0.
        patch_loss_g_value = 0.
        for (coarse_content, mask_content, Dmask_content), \
                (detailed_style, _, _) in zip(content_batch, style_batch):

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(coarse_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_style = torch.from_numpy(detailed_style).to(self.device).unsqueeze(0).unsqueeze(0)

            d_common_real = self.common_discriminator(voxel_style, is_training=True)
            z_tensor_g = self.discriminator_style(d_common_real)
            z_tensor_g = self.discriminator_style.pool(z_tensor_g)

            voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=True)

            d_common_fake = self.common_discriminator(voxel_fake, is_training=True)
            d_global_fake = self.discriminator_global(d_common_fake, is_training=False)
            d_style_fake = self.discriminator_style(d_common_fake, is_training=False)

            if self.use_wc:
                loss_g = - torch.sum(d_global_fake * Dmask_fake) / torch.sum(Dmask_fake)
            else:
                loss_g = torch.sum((d_global_fake - 1) ** 2 * Dmask_fake) / torch.sum(Dmask_fake)
            loss_g = loss_g / len_batch
            loss_g_value += loss_g.item()
            loss_g.backward(retain_graph=True)

            # d_common_real = self.common_discriminator(voxel_style, is_training=True)
            d_style_real = self.discriminator_style(d_common_real, is_training=False)

            real_patches, fake_patches = get_random_non_cube_pair_patches(d_style_real,
                                                                          d_style_fake,
                                                                          stride_factor=self.stride_factor,
                                                                          num_pairs=self.num_pairs)
            real_patches = self.discriminator_style.pool(real_patches).view(-1, self.z_dim)
            fake_patches = self.discriminator_style.pool(fake_patches).view(-1, self.z_dim)

            dist_real_fake = F.pairwise_distance(real_patches, fake_patches, 2)

            patch_loss_g = torch.mean(dist_real_fake) * self.param_alpha
            patch_loss_g = patch_loss_g / len_batch
            patch_loss_g_value += patch_loss_g.item()
            patch_loss_g.backward(retain_graph=False)

        if writer:
            common_disc_grads = plot_grad_flow(self.common_discriminator.named_parameters())
            disc_style_grads = plot_grad_flow(self.discriminator_style.named_parameters())
            gen_grads = plot_grad_flow(self.generator.named_parameters())
            writer.add_figure('common_disc_grads', common_disc_grads, iter_counter)
            writer.add_figure('disc_style_grads', disc_style_grads, iter_counter)
            writer.add_figure('gen_grads', gen_grads, iter_counter)

        self.optimizer_g.step()
        self.optimizer_d_common.step()
        self.optimizer_d_style.step()

        return loss_g_value, patch_loss_g_value

    def eval_step(self, content_batch, style_batch):
        len_batch = len(content_batch)

        loss_d_real_value = 0.
        loss_d_fake_value = 0.
        patch_loss_d_real_value = 0.
        patch_loss_d_fake_value = 0.
        loss_g_value = 0.
        patch_loss_g_value = 0.
        loss_r_value = 0.

        style_batch_vectors = []

        for (coarse_content, mask_content, Dmask_content), \
                (coarse_style, detailed_style, mask_style, Dmask_style, detailed_other_style) in zip(content_batch, style_batch):

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(coarse_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_style = torch.from_numpy(detailed_style).to(self.device).unsqueeze(0).unsqueeze(0)
            Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

            voxel_other_style = torch.from_numpy(detailed_other_style).to(self.device).unsqueeze(0).unsqueeze(0)

            d_common_real = self.common_discriminator(voxel_style, is_training=False)
            z_tensor_g = self.discriminator_style(d_common_real, is_training=False)
            z_tensor_g = self.discriminator_style.pool(z_tensor_g)
            voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)
            voxel_fake = voxel_fake.detach()

            d_global_real = self.discriminator_global(d_common_real, is_training=True)

            loss_d_real = torch.sum((d_global_real - 1) ** 2 * Dmask_style) / torch.sum(Dmask_style)
            loss_d_real = loss_d_real / len_batch
            loss_d_real_value += loss_d_real.item()

            d_common_fake = self.common_discriminator(voxel_fake, is_training=True)
            d_global_fake = self.discriminator_global(d_common_fake, is_training=True)

            loss_d_fake = torch.sum(d_global_fake ** 2 * Dmask_fake) / torch.sum(Dmask_fake)
            loss_d_fake = loss_d_fake / len_batch
            loss_d_fake_value += loss_d_fake.item()

            loss_g = torch.sum((d_global_fake - 1) ** 2 * Dmask_fake) / torch.sum(Dmask_fake)
            loss_g = loss_g / len_batch
            loss_g_value += loss_g.item()

            d_style_real = self.discriminator_style(d_common_real, is_training=True)
            d_style_fake = self.discriminator_style(d_common_fake, is_training=True)
            d_common_other = self.common_discriminator(voxel_other_style, is_training=True)
            d_style_other = self.discriminator_style(d_common_other, is_training=True)

            anchors, positives, negatives = get_random_non_cube_triplet_patches(d_style_real,
                                                                                d_style_other,
                                                                                stride_factor=self.stride_factor,
                                                                                num_triplets=self.num_triplets)
            anchors = self.discriminator_style.pool(anchors).view(-1, self.z_dim)
            positives = self.discriminator_style.pool(positives).view(-1, self.z_dim)
            negatives = self.discriminator_style.pool(negatives).view(-1, self.z_dim)

            patch_loss_d_real = self.triplet_criterion(anchors, positives, negatives) * self.param_gamma
            patch_loss_d_real = patch_loss_d_real / len_batch
            patch_loss_d_real_value += patch_loss_d_real.item()

            anchors, positives, negatives = get_random_non_cube_triplet_patches(d_style_real,
                                                                                d_style_fake,
                                                                                stride_factor=self.stride_factor,
                                                                                num_triplets=self.num_triplets)
            anchors = self.discriminator_style.pool(anchors).view(-1, self.z_dim)
            positives = self.discriminator_style.pool(positives).view(-1, self.z_dim)
            negatives = self.discriminator_style.pool(negatives).view(-1, self.z_dim)

            patch_loss_d_fake = self.triplet_criterion(anchors, positives, negatives) * self.param_delta
            patch_loss_d_fake = patch_loss_d_fake / len_batch
            patch_loss_d_fake_value += patch_loss_d_fake.item()

            real_patches, fake_patches = get_random_non_cube_pair_patches(d_style_real,
                                                                          d_style_fake,
                                                                          stride_factor=self.stride_factor,
                                                                          num_pairs=self.num_pairs)
            real_patches = self.discriminator_style.pool(real_patches).view(-1, self.z_dim)
            fake_patches = self.discriminator_style.pool(fake_patches).view(-1, self.z_dim)

            dist_real_fake = F.pairwise_distance(real_patches, fake_patches, 2)

            patch_loss_g = torch.mean(dist_real_fake) * self.param_alpha
            patch_loss_g = patch_loss_g / len_batch
            patch_loss_g_value += patch_loss_g.item()

            mask_style = torch.from_numpy(mask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()
            coarse_style = torch.from_numpy(coarse_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

            fake = self.generator(coarse_style, z_tensor_g, mask_style, is_training=True)

            loss_r = torch.mean((voxel_style - fake) ** 2) * self.param_beta
            loss_r = loss_r / len_batch
            loss_r_value += loss_r.item()

            style_batch_vectors.append(z_tensor_g[:,:,0,0,0].detach().cpu().numpy())

        style_batch_vectors = np.vstack(style_batch_vectors)
        return loss_d_real_value, loss_d_fake_value, patch_loss_d_real_value, patch_loss_d_fake_value,\
               loss_g_value, patch_loss_g_value, \
               loss_r_value, style_batch_vectors

    def train(self, config):

        del self.dset
        self.dset = self.style_set
        self.dataset_len = len(self.dset)

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

            self.common_discriminator.train()
            self.discriminator_global.train()
            self.discriminator_style.train()
            self.generator.train()

            self.dset.change_file_idx(epoch)
            self.style_set.change_file_idx(epoch)

            for idx in range(0, epoch_size-config.batch_size, config.batch_size):
                iter_counter += 1

                content_batch = []
                style_batch = []
                for b_idx in range(config.batch_size):

                    content_idx = batch_index_list[idx+b_idx]

                    possible_style_indices = self.dset.get_file_indices_with_same_rot(content_idx)

                    style_idx = random.choice(possible_style_indices)
                    while style_idx == content_idx:
                        style_idx = random.choice(possible_style_indices)
                    other_style_idx = random.choice(possible_style_indices)
                    while other_style_idx in [style_idx, content_idx]:
                        other_style_idx = random.choice(possible_style_indices)

                    content_data_dict = self.dset.__getitem__(content_idx)
                    mask_content = content_data_dict['mask']
                    Dmask_content = content_data_dict['Dmask']
                    input_content = content_data_dict['input']

                    style_data_dict = self.style_set.__getitem__(style_idx)
                    voxel_style = style_data_dict['voxel_style']
                    Dmask_style = style_data_dict['Dmask_style']

                    other_style_data_dict = self.style_set.__getitem__(other_style_idx)
                    voxel_other_style = other_style_data_dict['voxel_style']

                    content_batch.append((input_content, mask_content, Dmask_content))
                    style_batch.append((voxel_style, Dmask_style, voxel_other_style))

                loss_d_real_values = []
                loss_d_fake_values = []
                patch_loss_d_real_values = []
                patch_loss_d_fake_values = []
                for dstep in range(config.d_steps):  # notice that generator is not reused in each dstep as its not updated
                    loss_d_real_value, loss_d_fake_value, patch_loss_d_real_value, patch_loss_d_fake_value = \
                        self.discriminator_step(content_batch, style_batch,
                                                train_writer if iter_counter % config.log_iter == 0 else None,
                                                iter_counter)
                    loss_d_real_values.append(loss_d_real_value)
                    patch_loss_d_real_values.append(patch_loss_d_real_value)
                    patch_loss_d_fake_values.append(patch_loss_d_fake_value)
                    loss_d_fake_values.append(loss_d_fake_value)
                loss_d_real_value = np.mean(loss_d_real_values)
                loss_d_fake_value = np.mean(loss_d_fake_values)
                patch_loss_d_real_value = np.mean(patch_loss_d_real_values)
                patch_loss_d_fake_value = np.mean(patch_loss_d_fake_values)

                loss_r_values = []
                r_steps = self.r_steps if iter_counter < 5000 // config.batch_size else 1  # means after 2 epochs in chairs
                for step in range(r_steps):
                    real_style_batch = []
                    for style_idx in range(config.batch_size):
                        qxp = np.random.randint(self.styleset_len)
                        style_data_dict_2 = self.style_set.__getitem__(qxp)
                        voxel_style_2 = style_data_dict_2['voxel_style']
                        mask_style_2 = style_data_dict_2['mask']
                        input_style_2 = style_data_dict_2['input']
                        real_style_batch.append((input_style_2, voxel_style_2, mask_style_2))

                    loss_r_value = self.reconstruction_step(real_style_batch,
                                                            train_writer if iter_counter % config.log_iter == 0 else None,
                                                            iter_counter)
                    loss_r_values.append(loss_r_value)
                loss_r_value = np.mean(loss_r_values)

                loss_g_values = []
                patch_loss_g_values = []
                for gstep in range(self.g_steps):
                    loss_g_value, patch_loss_g_value = self.generator_step(content_batch, style_batch,
                                                                           train_writer if iter_counter % config.log_iter == 0 else None,
                                                                           iter_counter)
                    loss_g_values.append(loss_g_value)
                    patch_loss_g_values.append(patch_loss_g_value)
                loss_g_value = np.mean(loss_g_values)
                patch_loss_g_value = np.mean(patch_loss_g_values)

                if iter_counter % config.log_iter == 0:
                    print("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, "
                          "patch_loss_d_real: %.6f, patch_loss_d_fake: %.6f, "
                          "loss_r: %.6f, loss_g: %.6f, patch_loss_g: %.6f" % (
                        epoch, training_epoch, time.time() - start_time,
                        loss_d_real_value, loss_d_fake_value, patch_loss_d_real_value, patch_loss_d_fake_value,
                        loss_r_value, loss_g_value, patch_loss_g_value))
                    self.log.debug("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, "
                          "patch_loss_d_real: %.6f, patch_loss_d_fake: %.6f, "
                          "loss_r: %.6f, loss_g: %.6f, patch_loss_g: %.6f" % (
                        epoch, training_epoch, time.time() - start_time,
                        loss_d_real_value, loss_d_fake_value, patch_loss_d_real_value, patch_loss_d_fake_value,
                        loss_r_value, loss_g_value, patch_loss_g_value))
                    train_writer.add_scalar('loss_d_real', loss_d_real_value, iter_counter * config.batch_size)
                    train_writer.add_scalar('loss_d_fake', loss_d_fake_value, iter_counter * config.batch_size)
                    train_writer.add_scalar(f'patch_loss_d_real', patch_loss_d_real_value, iter_counter * config.batch_size)
                    train_writer.add_scalar(f'patch_loss_d_fake', patch_loss_d_fake_value, iter_counter * config.batch_size)
                    train_writer.add_scalar(f'loss_r', loss_r_value, iter_counter * config.batch_size)
                    train_writer.add_scalar('loss_g', loss_g_value, iter_counter * config.batch_size)
                    train_writer.add_scalar(f'patch_loss_g', patch_loss_g_value, iter_counter * config.batch_size)

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
        patch_loss_d_real = []
        patch_loss_d_fake = []
        loss_r = []
        patch_loss_g = []

        for idx in range(0, val_size - config.batch_size, config.batch_size):
            content_batch = []
            style_batch = []
            for b_idx in range(config.batch_size):
                dxb = idx + b_idx

                style_idx = np.random.randint(self.styleset_len)
                other_style_idx = np.random.randint(self.styleset_len)
                while other_style_idx == style_idx:
                    other_style_idx = np.random.randint(self.styleset_len)

                content_data_dict = self.vset.__getitem__(dxb)
                mask_content = content_data_dict['mask']
                Dmask_content = content_data_dict['Dmask']
                input_content = content_data_dict['input']

                style_data_dict = self.style_set.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']
                Dmask_style = style_data_dict['Dmask_style']
                coarse_style = style_data_dict['input']
                mask_style = style_data_dict['mask']

                other_style_data_dict = self.style_set.__getitem__(other_style_idx)
                voxel_other_style = other_style_data_dict['voxel_style']

                content_batch.append((input_content, mask_content, Dmask_content))
                style_batch.append((coarse_style, voxel_style, mask_style, Dmask_style, voxel_other_style))

            loss_d_real_value, loss_d_fake_value, patch_loss_d_real_value, patch_loss_d_fake_value, \
                loss_g_value, patch_loss_g_value, \
                loss_r_value, style_batch_vectors = self.eval_step(content_batch, style_batch)

            loss_d_real.append(loss_d_real_value)
            loss_d_fake.append(loss_d_fake_value)
            loss_g.append(loss_g_value)
            patch_loss_d_real.append(patch_loss_d_real_value)
            patch_loss_d_fake.append(patch_loss_d_fake_value)
            patch_loss_g.append(patch_loss_g_value)
            loss_r.append(loss_r_value)

        loss_d_real = np.mean(loss_d_real)
        loss_d_fake = np.mean(loss_d_fake)
        loss_g = np.mean(loss_g)
        patch_loss_d_real = np.mean(patch_loss_d_real)
        patch_loss_d_fake = np.mean(patch_loss_d_fake)
        patch_loss_g = np.mean(patch_loss_g)
        loss_r = np.mean(loss_r)
        print("Val Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, "
                          "patch_loss_d_real: %.6f, patch_loss_d_fake: %.6f, "
                          "loss_r: %.6f, loss_g: %.6f, patch_loss_g: %.6f" % (
                        epoch, training_epoch, time.time() - start_time,
                        loss_d_real, loss_d_fake, patch_loss_d_real, patch_loss_d_fake,
                        loss_r, loss_g, patch_loss_g))
        self.log.debug("Val Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, "
                          "patch_loss_d_real: %.6f, patch_loss_d_fake: %.6f, "
                          "loss_r: %.6f, loss_g: %.6f, patch_loss_g: %.6f" % (
                        epoch, training_epoch, time.time() - start_time,
                        loss_d_real, loss_d_fake, patch_loss_d_real, patch_loss_d_fake,
                        loss_r, loss_g, patch_loss_g))
        val_writer.add_scalar('loss_d_real', loss_d_real, iter_counter * config.batch_size)
        val_writer.add_scalar('loss_d_fake', loss_d_fake, iter_counter * config.batch_size)
        val_writer.add_scalar('loss_g', loss_g, iter_counter * config.batch_size)
        val_writer.add_scalar(f'patch_loss_d_real', patch_loss_d_real, iter_counter * config.batch_size)
        val_writer.add_scalar(f'patch_loss_d_fake', patch_loss_d_fake, iter_counter * config.batch_size)
        val_writer.add_scalar(f'patch_loss_g', patch_loss_g, iter_counter * config.batch_size)
        val_writer.add_scalar(f'loss_r', loss_r, iter_counter * config.batch_size)

        heatmap_fig = plot_matrix(style_batch_vectors, show=False)
        val_writer.add_figure('some_styles', heatmap_fig, iter_counter * config.batch_size)
