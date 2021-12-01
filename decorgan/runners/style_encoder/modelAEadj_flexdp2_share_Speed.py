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
from runners.common_dp2 import IM_AE_ADJ_SHARE_COMMON
from utils.io_helper import setup_logging
from utils.matplotlib_utils import plot_matrix

from utils import *
from modelAE_GD import *


class IM_AE(IM_AE_ADJ_SHARE_COMMON):

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
        self.common_discriminator = common_discriminator_2(self.d_dim)
        self.common_discriminator.to(self.device)

        self.discriminator_global = discriminator_part_global_plausibility_2(self.d_dim)
        self.discriminator_global.to(self.device)

        self.discriminator_style = discriminator_part_style_plausibility_2(self.d_dim, self.styleset_len)
        self.discriminator_style.to(self.device)

        self.style_encoder_part = style_encoder_part_encode(self.d_dim, self.z_dim)
        self.style_encoder_part.to(self.device)

        sigmoid = False
        if config.recon_loss == "BCE":
            sigmoid = True

        if self.input_size == 64 and self.output_size == 256:
            self.generator = generator_allstyles(self.g_dim, self.z_dim, sigmoid)
        elif self.input_size == 32 and self.output_size == 128:
            self.generator = generator_halfsize_allstyles(self.g_dim, self.z_dim, sigmoid)
        elif self.input_size == 32 and self.output_size == 256:
            self.generator = generator_halfsize_x8_allstyles(self.g_dim, self.z_dim, sigmoid)
        elif self.input_size == 16 and self.output_size == 128:
            self.generator = generator_halfsize_x8_allstyles(self.g_dim, self.z_dim, sigmoid)
        self.generator.to(self.device)

        self.optimizer_d_common = torch.optim.Adam(self.common_discriminator.parameters(), lr=config.lr)
        self.optimizer_d_global = torch.optim.Adam(self.discriminator_global.parameters(), lr=config.lr)
        self.optimizer_d_style = torch.optim.Adam(self.discriminator_style.parameters(), lr=config.lr)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr)
        self.optimizer_se = torch.optim.Adam(self.style_encoder_part.parameters(), lr=config.lr)

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

            self.common_discriminator.train()
            self.discriminator_global.train()
            self.discriminator_style.train()
            self.generator.train()
            self.style_encoder_part.train()

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

                d_common_real = self.common_discriminator(voxel_style, is_training=False)
                z_tensor_g = self.style_encoder_part(d_common_real, is_training=False)
                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)
                voxel_fake = voxel_fake.detach()

                #D step
                for dstep in range(config.d_steps):  # notice that generator is not reused in each dstep as its not updated

                    self.common_discriminator.zero_grad()
                    self.discriminator_global.zero_grad()
                    self.discriminator_style.zero_grad()

                    d_common_real = self.common_discriminator(voxel_style, is_training=True)
                    d_global_real = self.discriminator_global(d_common_real, is_training=True)
                    d_style_real = self.discriminator_style(d_common_real, is_training=True)

                    loss_d_real = (torch.sum(
                        (d_style_real[:, z_vector_style_idx:z_vector_style_idx + 1] - 1) ** 2 * Dmask_style) +
                     torch.sum((d_global_real - 1) ** 2 * Dmask_style)) / torch.sum(Dmask_style)
                    loss_d_real_value = loss_d_real.data # to be sure graph will be free afterwards..
                    loss_d_real.backward()  # no retain graph needed as only loss_d_real depends on voxel_style

                    d_common_fake = self.common_discriminator(voxel_fake, is_training=True)
                    d_global_fake = self.discriminator_global(d_common_fake, is_training=True)
                    d_style_fake = self.discriminator_style(d_common_fake, is_training=True)
                    loss_d_fake = (torch.sum((d_style_fake[:, z_vector_style_idx:z_vector_style_idx + 1]) ** 2 * Dmask_fake) +
                                   torch.sum((d_global_fake) ** 2 * Dmask_fake)) / torch.sum(Dmask_fake)
                    loss_d_fake_value = loss_d_fake.data
                    loss_d_fake.backward()  # no retain graph needed as only loss_d_fake depends on voxel_fake

                    self.optimizer_d_common.step()
                    self.optimizer_d_global.step()
                    self.optimizer_d_style.step()

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

                    self.common_discriminator.zero_grad()
                    self.style_encoder_part.zero_grad()
                    self.generator.zero_grad()

                    d_common_real_2 = self.common_discriminator(voxel_style_2, is_training=True)
                    z_tensor2_g = self.style_encoder_part(d_common_real_2, is_training=True)
                    voxel_fake = self.generator(input_style_2, z_tensor2_g, mask_style_2, is_training=True)

                    if config.recon_loss == 'MSE':
                        loss_r = torch.mean((voxel_style_2 - voxel_fake) ** 2) * self.param_beta
                    else:
                        loss_r = self.bce_loss(voxel_fake, voxel_style_2) * self.param_beta
                    loss_r_value = loss_r.data
                    loss_r.backward()
                    self.optimizer_g.step()
                    self.optimizer_d_common.step()
                    self.optimizer_se.step()

                # G step
                for gstep in range(self.g_steps):
                    self.generator.zero_grad()
                    self.common_discriminator.zero_grad()
                    self.discriminator_style.zero_grad()
                    self.style_encoder_part.zero_grad()

                    d_common_real = self.common_discriminator(voxel_style, is_training=True)
                    z_tensor_g = self.style_encoder_part(d_common_real)
                    voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=True)

                    d_common_fake = self.common_discriminator(voxel_fake, is_training=True)
                    d_global_fake = self.discriminator_global(d_common_fake, is_training=False)
                    d_style_fake = self.discriminator_style(d_common_fake, is_training=False)

                    loss_g = (torch.sum((d_style_fake[:,
                                         z_vector_style_idx:z_vector_style_idx + 1] - 1) ** 2 * Dmask_fake) * self.param_alpha + torch.sum(
                        (d_global_fake - 1) ** 2 * Dmask_fake)) / torch.sum(Dmask_fake)
                    loss_g_value = loss_g.data  # to be sure mem will be free
                    loss_g.backward(retain_graph=True)  # no retain graph needed as no other loss depends on these tensors

                    self.optimizer_g.step()
                    self.optimizer_d_common.step()
                    self.optimizer_d_style.step()
                    self.optimizer_se.step()

                if iter_counter % config.log_iter == 0:
                    print(
                        "Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                            epoch, training_epoch, time.time() - start_time, loss_d_real_value, loss_d_fake_value,
                            loss_r_value, loss_g_value))
                    self.log.debug(
                        "Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                            epoch, training_epoch, time.time() - start_time, loss_d_real_value, loss_d_fake_value,
                            loss_r_value, loss_g_value))
                    train_writer.add_scalar('loss_d_real', loss_d_real_value, iter_counter)
                    train_writer.add_scalar('loss_d_fake', loss_d_fake_value, iter_counter)
                    train_writer.add_scalar('loss_r', loss_r_value, iter_counter)
                    train_writer.add_scalar('loss_g', loss_g_value, iter_counter)

                    with torch.no_grad():  # to optimize performance as it sets require_grad temporarily to false
                        self.visualise(config.sample_dir, f"iter{iter_counter}")

                if iter_counter % config.save_iter == 0:
                    self.save(iter_counter)

            epoch += 1
            print(
                "Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                    epoch, training_epoch, time.time() - start_time, loss_d_real_value, loss_d_fake_value,
                    loss_r_value, loss_g_value))
            self.log.debug(
                "Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f" % (
                    epoch, training_epoch, time.time() - start_time, loss_d_real_value, loss_d_fake_value,
                    loss_r_value, loss_g_value))
            train_writer.add_scalar('loss_d_real', loss_d_real_value, iter_counter)
            train_writer.add_scalar('loss_d_fake', loss_d_fake_value, iter_counter)
            train_writer.add_scalar('loss_r', loss_r_value, iter_counter)
            train_writer.add_scalar('loss_g', loss_g_value, iter_counter)

            with torch.no_grad():
                self.visualise(config.sample_dir, f"epoch{epoch}")
            self.save(iter_counter)

            if epoch % 1 == 0:
                cv2.imwrite(config.sample_dir + "/" + str(epoch) + "_0.png", self.imgout_0)

        # if finish, save
        self.save(iter_counter)

    def visualise(self, sample_dir, epoch):

        self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)

        for style_idx in range(0, 5):
            # _, _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
            _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
            self.imgout_0[0:self.real_size,
                          (style_idx+1)*self.real_size:(style_idx+2)*self.real_size] = detailed_full_style_img

        for content_idx in range(5):

            # coarse_content_img, _, _ = self.coarse_detailed_full_plots(content_idx, content=True)
            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx, content=True)
            self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                          0:self.real_size] = coarse_content_img

            content_data_dict = self.dset.__getitem__(content_idx)
            mask_content = content_data_dict['mask']
            input_content = content_data_dict['input']
            xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_idx in range(0, 5):

                style_data_dict = self.style_set.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']

                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                d_common_real = self.common_discriminator(voxel_style, is_training=False)
                z_tensor_g = self.style_encoder_part(d_common_real)
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                # _, detailed_gen_smooth_img, _ = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                (style_idx+1)*self.real_size: (style_idx+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)

    def save(self, iter_counter):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(iter_counter)+".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
        #delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        #save checkpoint
        torch.save({
                    'generator': self.generator.state_dict(),
                    'discriminator_global': self.discriminator_global.state_dict(),
                    'discriminator_style': self.discriminator_style.state_dict(),
                    'common_discriminator': self.common_discriminator.state_dict(),
                    'style_encoder_part': self.style_encoder_part.state_dict(),
                    }, save_dir)
        #update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        #write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer]+"\n")
        fout.close()

    def load(self, gpu):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            iter_counter = os.path.basename(model_dir).split("-")[-1].replace(".pth", "")
            checkpoint = torch.load(model_dir, map_location=f'cuda')
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator_global.load_state_dict(checkpoint['discriminator_global'])
            self.discriminator_style.load_state_dict(checkpoint['discriminator_style'])
            self.common_discriminator.load_state_dict(checkpoint['common_discriminator'])
            self.style_encoder_part.load_state_dict(checkpoint['style_encoder_part'])
            print(" [*] Load SUCCESS")
            self.log.debug(" [*] Load SUCCESS")
            return int(iter_counter)
        else:
            print(" [!] Load failed...")
            self.log.debug(" [!] Load failed...")
            return False

    def test_fig_3(self, config):

        style_indices = [int(i) for i in config.style_indices.split(", ")]

        self.voxel_renderer.use_gpu(config.gpu)

        if not self.load(config.gpu): exit(-1)
        os.makedirs(config.test_fig_3_dir, exist_ok=True)

        self.imgout_0 = np.full([self.real_size*(self.dataset_len+1),
                                 self.real_size*(len(style_indices)+1)], 255, np.uint8)

        for content_idx in range(self.dataset_len):
            # coarse_content_img, _, _ = self.coarse_detailed_full_plots(content_idx, content=True)
            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx, content=True)
            self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size,
                          0:self.real_size] = coarse_content_img

        for style_idx in range(self.styleset_len):
            if style_idx not in style_indices:
                continue

            # _, _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
            _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
            self.imgout_0[0:self.real_size,
                          (style_indices.index(style_idx) + 1)*self.real_size: (style_indices.index(style_idx) + 2)*self.real_size] = detailed_full_style_img

            voxel_style = self.style_set.__getitem__(style_idx)['voxel_style']
            voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

            d_common_real = self.common_discriminator(voxel_style, is_training=False)
            z_tensor_g = self.style_encoder_part(d_common_real)

            for content_idx in range(self.dataset_len):

                content_data_dict = self.dset.__getitem__(content_idx)
                mask_content = content_data_dict['mask']
                input_content = content_data_dict['input']
                xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

                mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                # _, detailed_gen_smooth_img, _ = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)

                self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size,
                              (style_indices.index(style_idx)+1)*self.real_size:(style_indices.index(style_idx)+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(config.test_fig_3_dir,f"fig3.png"), self.imgout_0)

    def export(self, config):

        raise Exception()
        if not self.load(config.gpu): exit(-1)

        with torch.no_grad():

            out_discr_common_global_max = os.path.join(config.export_dir, "discr_common_global/max")
            out_discr_global_max = os.path.join(config.export_dir, "discr_global/max")
            out_discr_common_style_max = os.path.join(config.export_dir, "discr_common_style/max")
            out_discr_style_max = os.path.join(config.export_dir, "discr_style/max")
            out_discr_all_max = os.path.join(config.export_dir, "discr_all/max")

            for i in range(self.dataset_len):

                inp_file = self.dset.get_model_name(i)
                try:
                    building = inp_file.split("/")[0]
                    component = inp_file.split("/")[1].replace("style_mesh_", "")
                except:
                    building = inp_file
                    component = "whole"

                if self.dset.last_cache_percent != 100:
                    data_dict = self.dset.get_without_cache(i)
                else:
                    data_dict = self.dset.__getitem__(i)

                detail_input = data_dict['voxel_style']

                detail_input = torch.from_numpy(detail_input).to(self.device).unsqueeze(0).unsqueeze(0)

                d_common = self.common_discriminator(detail_input, is_training=True)
                d_global = self.discriminator_global.layer(d_common, is_training=False)
                d_style = self.discriminator_style.layer(d_common,)
                d_common = self.common_discriminator.pool(d_common).squeeze(2).squeeze(2).squeeze(2)

                out = torch.cat([d_common.T, d_global.T])
                out_style_file = os.path.join(out_discr_common_global_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, out.cpu().numpy().reshape((-1)))

                out = torch.cat([d_common.T, d_style.T])
                out_style_file = os.path.join(out_discr_common_style_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, out.cpu().numpy().reshape((-1)))

                out_style_file = os.path.join(out_discr_style_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, d_style.cpu().numpy().reshape((-1)))

                out_style_file = os.path.join(out_discr_global_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, d_global.cpu().numpy().reshape((-1)))

                out = torch.cat([d_common.T, d_global.T, d_style.T])
                out_style_file = os.path.join(out_discr_all_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, out.cpu().numpy().reshape((-1)))
