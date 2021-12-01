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

from utils import *
from modelAE_GD import *
from dataset import *
from utils.io_helper import setup_logging
from utils.pytorch3d_vis import CustomDefinedViewMeshRenderer
from utils.matplotlib_utils import render_result, render_example, render_views, plot_matrix
import mcubes
from utils.open3d_utils import TriangleMesh
from utils.open3d_render import render_geometries
from utils import CameraJsonPosition
from PIL import Image
import matplotlib.pyplot as plt_original
from utils.nt_xent import original_nt_xent


# cdvr = CustomDefinedViewMeshRenderer(4)


def get_building_from_filename(file):
    return os.path.basename(os.path.dirname(os.path.dirname(file)))


def get_element_from_filename(file):
    return os.path.basename(os.path.dirname(file)).split("__")[0].split("_")[-1]


# def img_of_input(voxel_model_file):
#     with open(voxel_model_file, 'rb') as fin:
#         voxel_model_512 = binvox_rw.read_as_3d_array(fin, fix_coords=True).data.astype(np.uint8)
#     vertices, triangles = mcubes.marching_cubes(voxel_model_512, 0.5)
#     vertices = normalize_vertices(vertices)
#     mcubes_in = cdvr(verts=vertices, triangles=triangles.astype(int))
#     return mcubes_in


class IM_AE(object):

    def coarse_detailed_full_plots(self, i):
        data_dict = self.dset.__getitem__(i)
        xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
        voxel_style = data_dict['voxel_style']
        tmp, _ = self.dset.get_more(i)

        # tmpvox = self.recover_voxel(voxel_style, xmin, xmax, ymin, ymax, zmin, zmax)
        # if not self.local:
        #     img_det_smooth1 = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, 0)
        #     img_det_smooth2 = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, 6)
        #     img_det_smooth = render_views([img_det_smooth1, img_det_smooth2])
        # else:
        #     vertices, triangles = mcubes.marching_cubes(tmpvox, self.sampling_threshold)
        #     vertices = normalize_vertices(vertices)
        #     m_smooth = TriangleMesh(vertices, triangles)
        #     m_smooth.compute_vertex_normals()
        #     img_det_smooth = render_geometries([m_smooth], camera_json=CameraJsonPosition, out_img=True)

        tmp_mask_exact = self.get_voxel_mask_exact(tmp)
        tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
        if not self.local:
            img_coa1 = self.voxel_renderer.render_img(tmpvox, 0, 0)
            img_coa2 = self.voxel_renderer.render_img(tmpvox, 0, 6)
            img_coa = render_views([img_coa1, img_coa2])
        else:
            vertices, triangles = mcubes.marching_cubes(tmpvox, self.sampling_threshold)
            vertices = normalize_vertices(vertices)
            m_coa = TriangleMesh(vertices, triangles)
            m_coa.compute_vertex_normals()
            img_coa = render_geometries([m_coa], camera_json=CameraJsonPosition, out_img=True)
            img_coa = Image.fromarray(np.uint8(np.asarray(img_coa) * 255))
            img_coa = np.asarray(img_coa.resize((self.real_size, self.real_size)).convert('L'))

        tmpvox = self.recover_voxel(tmp, xmin, xmax, ymin, ymax, zmin, zmax)
        if not self.local:
            img_det_full1 = self.voxel_renderer.render_img(tmpvox, 0, 0)
            img_det_full2 = self.voxel_renderer.render_img(tmpvox, 0, 6)
            img_det_full = render_views([img_det_full1, img_det_full2])
        else:
            vertices, triangles = mcubes.marching_cubes(tmpvox, 0)
            vertices = normalize_vertices(vertices)
            m_det_full = TriangleMesh(vertices, triangles)
            m_det_full.compute_vertex_normals()
            img_det_full = render_geometries([m_det_full], camera_json=CameraJsonPosition, out_img=True)
            img_det_full = Image.fromarray(np.uint8(np.asarray(img_det_full) * 255))
            img_det_full = np.asarray(img_det_full.resize((self.real_size, self.real_size)).convert('L'))

        # return img_coa, img_det_smooth, img_det_full
        return img_coa, img_det_full

    # def generation_plot(self, vox, xmin, xmax, ymin, ymax, zmin, zmax):
    #     tmpvox = self.recover_voxel(vox, xmin, xmax, ymin, ymax, zmin, zmax)
    #     if not self.local:
    #         img_smooth_det1 = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, 0)
    #         img_smooth_det2 = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, 6)
    #         img_smooth_det = render_views([img_smooth_det1, img_smooth_det2])
    #     else:
    #         vertices, triangles = mcubes.marching_cubes(tmpvox, self.sampling_threshold)
    #         if len(vertices) == 0:
    #             img_smooth_det = None
    #         else:
    #             vertices = normalize_vertices(vertices)
    #             m_smooth = TriangleMesh(vertices, triangles)
    #             m_smooth.compute_vertex_normals()
    #             img_smooth_det = render_geometries([m_smooth], camera_json=CameraJsonPosition, out_img=True)
    #
    #     if not self.local:
    #         img_full_det1 = self.voxel_renderer.render_img(tmpvox, 0, 0)
    #         img_full_det2 = self.voxel_renderer.render_img(tmpvox, 0, 6)
    #         img_full_det = render_views([img_full_det1, img_full_det2])
    #     else:
    #         vertices, triangles = mcubes.marching_cubes(tmpvox, 0)
    #         if len(vertices) == 0:
    #             img_full_det = None
    #         else:
    #             vertices = normalize_vertices(vertices)
    #             m_full_det = TriangleMesh(vertices, triangles)
    #             m_full_det.compute_vertex_normals()
    #             img_full_det = render_geometries([m_full_det], camera_json=CameraJsonPosition, out_img=True)
    #
    #     tmp_mask_exact = self.get_voxel_mask_exact(vox)
    #     tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
    #     if not self.local:
    #         img_coa1 = self.voxel_renderer.render_img(tmpvox, 0, 0)
    #         img_coa2 = self.voxel_renderer.render_img(tmpvox, 0, 6)
    #         img_coa = render_views([img_coa1, img_coa2])
    #     else:
    #         vertices, triangles = mcubes.marching_cubes(tmpvox, self.sampling_threshold)
    #         if len(vertices) == 0:
    #             img_coa = None
    #         else:
    #             vertices = normalize_vertices(vertices)
    #             m_coa = TriangleMesh(vertices, triangles)
    #             m_coa.compute_vertex_normals()
    #             img_coa = render_geometries([m_coa], camera_json=CameraJsonPosition, out_img=True)
    #
    #     return img_coa, img_smooth_det, img_full_det
    def generation_plot(self, vox, xmin, xmax, ymin, ymax, zmin, zmax):
        tmpvox = self.recover_voxel(vox, xmin, xmax, ymin, ymax, zmin, zmax)
        if not self.local:
            img_smooth_det1 = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, 0)
            img_smooth_det2 = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, 6)
            img_smooth_det = render_views([img_smooth_det1, img_smooth_det2])
        else:
            vertices, triangles = mcubes.marching_cubes(tmpvox, self.sampling_threshold)
            if len(vertices) == 0:
                img_smooth_det = np.ones((self.real_size, self.real_size)).astype(np.uint8) * 255
            else:
                vertices = normalize_vertices(vertices)
                m_smooth = TriangleMesh(vertices, triangles)
                m_smooth.compute_vertex_normals()
                img_smooth_det = render_geometries([m_smooth], camera_json=CameraJsonPosition, out_img=True)
                img_smooth_det = Image.fromarray(np.uint8(np.asarray(img_smooth_det) * 255))
                img_smooth_det = np.asarray(img_smooth_det.resize((self.real_size, self.real_size)).convert('L'))
        return img_smooth_det

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

        self.g_dim = 32
        self.d_dim = 32
        # self.z_dim = 8
        # self.z_dim = 256
        self.z_dim = 64
        self.s_dim = 16
        self.param_alpha = config.alpha
        self.param_beta = config.beta
        self.nt_xent_factor = config.nt_xent_factor
        self.style_batch = config.style_batch

        self.input_size = 16
        self.output_size = 128
        self.upsample_rate = 8

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

        self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)

        self.dset = BasicDataset(self.data_dir, self.datapath, config, self.log, filename=config.filename)
        self.dataset_len = len(self.dset)
        if config.buildings_dir is not None:
            self.bset = BuildingDataset(config.buildings_dir)
        else:
            self.bset = None

        if config.train:
            for i in range(16):
                # TODO: get_vox_from_binvox_1over2_return_small
                data_dict = self.dset.__getitem__(i)
                xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
                voxel_style = data_dict['voxel_style']

                tmp, _ = self.dset.get_more(i)

                img_y = i // 4
                img_x = (i % 4) * 2 + 1
                if img_y < 4:
                    tmpvox = self.recover_voxel(voxel_style, xmin, xmax, ymin, ymax, zmin, zmax)
                    self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                    img_x * self.real_size:(img_x + 1) * self.real_size] = self.voxel_renderer.render_img(tmpvox,
                                                                                                          self.sampling_threshold,
                                                                                                          self.render_view_id)
                img_y = i // 4
                img_x = (i % 4) * 2
                if img_y < 4:
                    tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                    tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                    self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                    img_x * self.real_size:(img_x + 1) * self.real_size] = self.voxel_renderer.render_img(tmpvox,
                                                                                                          self.sampling_threshold,
                                                                                                          self.render_view_id)


        if config.train: cv2.imwrite(config.sample_dir + "/a_style_0.png", self.imgout_0)

        self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)

        if config.train:
            for i in range(16):
                # todo: get_vox_from_binvox_1over2_return_small
                data_dict = self.dset.__getitem__(i)
                xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

                tmp, _ = self.dset.get_more(i)

                img_y = i // 4
                img_x = (i % 4) * 2
                if img_y < 4:
                    tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                    tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                    self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                    img_x * self.real_size:(img_x + 1) * self.real_size] = self.voxel_renderer.render_img(tmpvox,
                                                                                                          self.sampling_threshold,
                                                                                                          self.render_view_id)

        if config.train: cv2.imwrite(config.sample_dir + "/a_content_0.png", self.imgout_0)

        # build model
        self.discriminator = discriminator(self.d_dim, 1)  # todo: try with given styles also
        self.discriminator.to(self.device)

        sigmoid = False
        if config.recon_loss == "BCE":
            sigmoid = True
        self.generator = generator_halfsize_x8_allstyles(self.g_dim, self.z_dim, sigmoid)
        self.generator.to(self.device)

        # self.style_encoder = style_encoder(self.s_dim, self.z_dim)
        # self.style_encoder = big_style_encoder(self.z_dim)
        self.style_encoder = style_encoder_64()
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

    def get_voxel_mask_exact(self, vox):
        # 256 -maxpoolk4s4- 64 -upsample- 256
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        # input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate,
                                         padding=0)
        # mask
        smallmask_tensor = F.interpolate(smallmaskx_tensor, scale_factor=self.upsample_rate, mode='nearest')
        # to numpy
        smallmask = smallmask_tensor.detach().cpu().numpy()[0, 0]
        smallmask = np.round(smallmask).astype(np.uint8)
        return smallmask

    def recover_voxel(self, vox, xmin, xmax, ymin, ymax, zmin, zmax):
        tmpvox = np.zeros([self.real_size, self.real_size, self.real_size], np.float32)
        xmin_, ymin_, zmin_ = (0, 0, 0)
        xmax_, ymax_, zmax_ = vox.shape
        xmin = xmin * self.upsample_rate - self.mask_margin
        xmax = xmax * self.upsample_rate + self.mask_margin
        ymin = ymin * self.upsample_rate - self.mask_margin
        ymax = ymax * self.upsample_rate + self.mask_margin
        if self.asymmetry:
            zmin = zmin * self.upsample_rate - self.mask_margin
        else:
            zmin = zmin * self.upsample_rate
            zmin_ = self.mask_margin
        zmax = zmax * self.upsample_rate + self.mask_margin
        if xmin < 0:
            xmin_ = -xmin
            xmin = 0
        if xmax > self.real_size:
            xmax_ = xmax_ + self.real_size - xmax
            xmax = self.real_size
        if ymin < 0:
            ymin_ = -ymin
            ymin = 0
        if ymax > self.real_size:
            ymax_ = ymax_ + self.real_size - ymax
            ymax = self.real_size
        if zmin < 0:
            zmin_ = -zmin
            zmin = 0
        if zmax > self.real_size:
            zmax_ = zmax_ + self.real_size - zmax
            zmax = self.real_size
        if self.asymmetry:
            tmpvox[xmin:xmax, ymin:ymax, zmin:zmax] = vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_]
        else:
            tmpvox[xmin:xmax, ymin:ymax, zmin:zmax] = vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_]
            if zmin * 2 - zmax - 1 < 0:
                tmpvox[xmin:xmax, ymin:ymax, zmin - 1::-1] = vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_]
            else:
                tmpvox[xmin:xmax, ymin:ymax, zmin - 1:zmin * 2 - zmax - 1:-1] = vox[xmin_:xmax_, ymin_:ymax_,
                                                                                zmin_:zmax_]
        return tmpvox

    def load(self):
        # load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            epoch = os.path.basename(model_dir).split("-")[-1].replace(".pth", "")
            checkpoint = torch.load(model_dir)
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.style_encoder.load_state_dict(checkpoint['style_encoder'])
            print(" [*] Load SUCCESS")
            self.log.debug(" [*] Load SUCCESS")
            return int(epoch)
        else:
            print(" [!] Load failed...")
            self.log.debug(" [!] Load failed...")
            return False

    def save(self, epoch):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path, self.checkpoint_name + "-" + str(epoch) + ".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.max_to_keep
        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        # save checkpoint
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
        }, save_dir)
        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        # write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer + self.max_to_keep - i) % self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer] + "\n")
        fout.close()

    @property
    def model_dir(self):
        return "ae"

    def train(self, config):

        if not config.debug:
            self.visualise_init(config.sample_dir)

        epoch_loaded = self.load()

        train_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(self.checkpoint_dir), "train_log"))

        start_time = time.time()
        training_epoch = config.epoch

        if not epoch_loaded:
            epoch_loaded = 0
            iter_counter = 0
        else:
            epoch_loaded = epoch_loaded + 1
            iter_counter = epoch_loaded * self.dataset_len

        batch_index_list = np.arange(self.dataset_len)

        epoch_size = self.dataset_len
        if config.debug:
            epoch_size = 10

        for epoch in range(epoch_loaded, training_epoch+1):
            np.random.shuffle(batch_index_list)

            self.discriminator.train()  # this is for dropout layers and batch normalization layers
            self.generator.train()
            self.style_encoder.train()

            for idx in range(epoch_size):
                iter_counter += 1

                # ready a fake image
                content_idx = batch_index_list[idx]
                content_data_dict = self.dset.__getitem__(content_idx)
                mask_content = content_data_dict['mask']
                Dmask_content = content_data_dict['Dmask']
                input_content = content_data_dict['input']

                mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                # use same styles during generator's and discriminator's steps
                style_indices = list(set(range(self.dataset_len)) - {content_idx})
                np.random.shuffle(style_indices)

                # D step
                d_step = 1
                for dstep in range(d_step):

                    self.discriminator.zero_grad()

                    loss_d_real_total = 0.
                    loss_d_fake_total = 0.
                    for style_idx in style_indices[0:self.style_batch]:
                        style_dict = self.dset.__getitem__(style_idx)
                        voxel_style = style_dict['voxel_style']
                        voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                        _, Dmask_style = self.dset.get_more(style_idx)
                        Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

                        D_out = self.discriminator(voxel_style, is_training=True)
                        loss_d_real = torch.sum((D_out - 1) ** 2 * Dmask_style) / torch.sum(Dmask_style)
                        loss_d_real = loss_d_real / self.style_batch
                        loss_d_real.backward()
                        loss_d_real_total = loss_d_real_total + loss_d_real.item()

                        z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1, -1, 1, 1, 1])
                        voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)
                        voxel_fake = voxel_fake.detach()  # probably unessesary since correct optimizer is called and
                        # corresponding gradients are set to zero at the start of the corresponding step

                        D_out = self.discriminator(voxel_fake, is_training=True)
                        loss_d_fake = torch.sum(D_out ** 2 * Dmask_fake) / torch.sum(Dmask_fake)
                        loss_d_fake = loss_d_fake / self.style_batch
                        loss_d_fake.backward()
                        loss_d_fake_total = loss_d_fake_total + loss_d_fake.item()

                    self.optimizer_d.step()

                # recon step
                # reconstruct style image
                r_step = 4 if iter_counter < 5000 else 1  # means after 2 epochs in chairs
                for rstep in range(r_step):
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

                    if config.recon_loss == 'MSE':
                        loss_r = torch.mean((voxel_style_2 - voxel_fake) ** 2) * self.param_beta
                    else:
                        loss_r = self.bce_loss(voxel_fake, voxel_style_2)*self.param_beta

                    loss_r.backward()
                    self.optimizer_g.step()
                    self.optimizer_se.step()
                    del voxel_style_2, mask_style_2, input_style_2  # free up some memory

                # G step (+style encoder in this step)
                g_step = 1
                for step in range(g_step):
                    self.style_encoder.zero_grad()
                    self.generator.zero_grad()

                    loss_g_init_total = 0.
                    all_styles = []
                    all_generated_styles = []
                    for style_idx in style_indices[0:self.style_batch]:
                        style_dict = self.dset.__getitem__(style_idx)
                        voxel_style = style_dict['voxel_style']
                        voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                        z_tensor_g = self.style_encoder(voxel_style, is_training=True)
                        voxel_fake = self.generator(input_fake, z_tensor_g.view([1, -1, 1, 1, 1]), mask_fake,
                                                    is_training=True)
                        voxel_fake_z_tensor_g = self.style_encoder(voxel_fake, is_training=True)
                        D_out = self.discriminator(voxel_fake, is_training=False)
                        loss_g_init = torch.sum((D_out - 1) ** 2 * Dmask_fake) * self.param_alpha / torch.sum(
                            Dmask_fake)
                        loss_g_init = loss_g_init / self.style_batch
                        loss_g_init.backward(retain_graph=True)  # calculate gradients on correct params,
                        # but don't delete anything as they will be used in loss_nt_xent.backward
                        loss_g_init_total = loss_g_init_total + loss_g_init.item()

                        all_styles.append(z_tensor_g)
                        all_generated_styles.append(voxel_fake_z_tensor_g)

                    all_generated_styles = torch.cat(all_generated_styles).view(-1, self.z_dim)
                    all_styles = torch.cat(all_styles).view(-1, self.z_dim)

                    loss_nt_xent = original_nt_xent(all_generated_styles, all_styles) * self.nt_xent_factor
                    loss_nt_xent.backward()  # calculate gradients on correct params

                    self.optimizer_g.step()  # update parameters of G and SE
                    self.optimizer_se.step()  # update parameters of G and SE

                if iter_counter % 500 == 0:
                    print(f"iter {iter_counter}")
                    train_writer.add_scalar('loss_d_real', loss_d_real_total, iter_counter)
                    train_writer.add_scalar('loss_d_fake', loss_d_fake_total, iter_counter)
                    train_writer.add_scalar('loss_r', loss_r.item(), iter_counter)
                    train_writer.add_scalar('loss_g_init', loss_g_init_total, iter_counter)
                    train_writer.add_scalar('loss_nt_xent', loss_nt_xent.item(), iter_counter)

                    heatmap_fig = plot_matrix(all_generated_styles.detach().cpu().numpy(), show=False)
                    train_writer.add_figure('all_generated_styles', heatmap_fig, iter_counter)
                    heatmap_fig = plot_matrix(all_styles.detach().cpu().numpy(), show=False)
                    train_writer.add_figure('all_styles', heatmap_fig, iter_counter)

            print("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g_init: %.6f "
                  "loss_nt_xent: %.6f" % (epoch, training_epoch, time.time() - start_time, loss_d_real_total,
                                          loss_d_fake_total, loss_r.item(), loss_g_init_total, loss_nt_xent.item()))
            self.log.debug("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, "
                           "loss_g_init: %.6f loss_nt_xent: %.6f"
                           % (epoch, training_epoch, time.time() - start_time, loss_d_real_total, loss_d_fake_total,
                              loss_r.item(), loss_g_init_total, loss_nt_xent.item()))
            train_writer.add_scalar('loss_d_real', loss_d_real_total, iter_counter)
            train_writer.add_scalar('loss_d_fake', loss_d_fake_total, iter_counter)
            train_writer.add_scalar('loss_r', loss_r.item(), iter_counter)
            train_writer.add_scalar('loss_g_init', loss_g_init_total, iter_counter)
            train_writer.add_scalar('loss_nt_xent', loss_nt_xent.item(), iter_counter)

            heatmap_fig = plot_matrix(all_generated_styles.detach().cpu().numpy(), show=False)
            train_writer.add_figure('all_generated_styles', heatmap_fig, iter_counter)
            heatmap_fig = plot_matrix(all_styles.detach().cpu().numpy(), show=False)
            train_writer.add_figure('all_styles', heatmap_fig, iter_counter)

            self.visualise(config.sample_dir, epoch)

            if epoch % self.save_epoch == 0:
                self.save(epoch)

        # if finish, save
        self.save(epoch)


    def visualise_init(self, sample_dir):

        for content_idx in range(5):

            content_reference_file = self.dset.get_reference_file(content_idx)

            # coarse_img, detailed_smooth_img, detailed_full_img = self.coarse_detailed_full_plots(content_idx)
            coarse_img, detailed_full_img = self.coarse_detailed_full_plots(content_idx)

            title = 'Content'
            if self.bset:
                title = f"{title} {get_element_from_filename(content_reference_file)}"

            render_example([coarse_img,
                           detailed_full_img],
                           save=os.path.join(sample_dir, f"content{content_idx}.png"),
                           title=title,
                           titles=['Coarse', 'Detailed Full'])

        for style_idx in range(3, 8):
            style_reference_file = self.dset.get_reference_file(style_idx)

            # coarse_img, detailed_smooth_img, detailed_full_img = self.coarse_detailed_full_plots(style_idx)
            coarse_img, detailed_full_img = self.coarse_detailed_full_plots(style_idx)

            title = 'Style'
            if self.bset:
                title = f"{title} {get_element_from_filename(style_reference_file)}"

            render_example([coarse_img,
                           detailed_full_img],
                           save=os.path.join(sample_dir, f"style{style_idx}.png"),
                           title=title,
                           titles=['Coarse', 'Detailed Full'])


    def visualise(self, sample_dir, epoch):

        self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)

        for style_idx in range(5, 10):
            _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx)
            self.imgout_0[0:self.real_size, (style_idx-5+1)*self.real_size:(style_idx+5)*self.real_size] = detailed_full_style_img

        for content_idx in range(5):

            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx)
            self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size, 0:self.real_size] = coarse_content_img

            content_data_dict = self.dset.__getitem__(content_idx)
            mask_content = content_data_dict['mask']
            input_content = content_data_dict['input']
            xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_idx in range(5, 10):

                style_data_dict = self.dset.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']

                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                z_tensor_g = self.style_encoder(voxel_style, is_training=False)
                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                # coarse_img, detailed_img_smooth, detailed_img_full = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                detailed_img_smooth = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,(style_idx-5+1)*self.real_size: (style_idx-5+2)*self.real_size] = detailed_img_smooth

        cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)

    def export(self, config):

        if not self.load(): exit(-1)

        self.dset = BasicDataset(self.data_dir, self.datapath, config, self.log, filename=config.filename)
        self.dataset_len =len(self.dset)

        with torch.no_grad():

            for i in range(self.dataset_len):

                inp_file = self.dset.files[i]
                # if not "RELIGIOUS" in inp_file:
                #     continue
                building = inp_file.split("/")[-3]
                component = inp_file.split("/")[-2].replace("style_mesh_", "")
                out_style_file = os.path.join(config.export_dir, "style", building, f"{component}.npy")
                out_content_file = os.path.join(config.export_dir, "content", building, f"{component}.npy")
                if os.path.exists(out_content_file) and os.path.exists(out_style_file):
                    continue

                try:
                    data_dict = self.dset.__getitem__(i)
                except:
                    continue

                detail_input = data_dict['voxel_style']

                detail_input = torch.from_numpy(detail_input).to(self.device).unsqueeze(0).unsqueeze(0)
                z_tensor_g = self.style_encoder(detail_input, is_training=False)

                os.makedirs(os.path.join(config.export_dir, "style", building), exist_ok=True)
                np.save(out_style_file, z_tensor_g.cpu().numpy().reshape((-1)))

                mask = data_dict['mask']
                input = data_dict['input']

                mask_fake = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(input).to(self.device).unsqueeze(0).unsqueeze(0).float()

                latent_g = self.generator.export(input_fake, z_tensor_g, mask_fake, is_training=False)
                latent_g = latent_g.detach().cpu().numpy()
                latent_g = np.max(latent_g, (0,2,3,4))
                os.makedirs(os.path.join(config.export_dir, "content", building), exist_ok=True)
                np.save(out_content_file, latent_g)

    def test_style_codes(self, config):

        self.voxel_renderer.use_gpu()

        if not self.load(): exit(-1)
        os.makedirs(config.style_codes_dir, exist_ok=True)

        max_num_of_styles = 64
        max_num_of_styles = min(max_num_of_styles, len(self.dset))

        style_codes = []
        with torch.no_grad():
            for style_idx in range(0, max_num_of_styles):
                style_data_dict = self.dset.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                z_tensor_g = self.style_encoder(voxel_style, is_training=False)
                style_codes.append(z_tensor_g.detach().cpu().numpy()[0, :, 0, 0, 0])
        style_codes = np.vstack(style_codes)
        # style_codes = (style_codes - np.mean(style_codes, axis=0)) / np.std(style_codes, axis=0)

        style_codes_img = plot_matrix(style_codes, as_img=True)
        cv2.imwrite(config.style_codes_dir + "/" + "style_codes.png", style_codes_img)

        embedded = TSNE(n_components=2, perplexity=16, learning_rate=10.0, n_iter=2000).fit_transform(style_codes)

        print("rendering...")
        img_size = 5000
        grid_size = 20
        if self.output_size == 128:
            cell_size = 140
        elif self.output_size == 256:
            cell_size = 180
        plt = np.full([img_size + self.real_size, img_size + self.real_size], 255, np.uint8)
        plt_grid = np.full([grid_size * cell_size + (self.real_size - cell_size),
                            grid_size * cell_size + (self.real_size - cell_size)], 255, np.uint8)
        occ_grid = np.zeros([grid_size, grid_size], np.uint8)

        x_max = np.max(embedded[:, 0])
        x_min = np.min(embedded[:, 0])
        y_max = np.max(embedded[:, 1])
        y_min = np.min(embedded[:, 1])
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        scalex = (x_max - x_min) * 1.05
        scaley = (y_max - y_min) * 1.05
        embedded[:, 0] = ((embedded[:, 0] - x_mid) / scalex + 0.5) * img_size
        embedded[:, 1] = ((embedded[:, 1] - y_mid) / scaley + 0.5) * img_size

        for i in range(max_num_of_styles):
            data_dict = self.dset.__getitem__(i)
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
            voxel_style = data_dict['voxel_style']
            tmp, _ = self.dset.get_more(i)
            tmpvox = self.recover_voxel(voxel_style, xmin, xmax, ymin, ymax, zmin, zmax)
            # rendered_view = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvox, self.sampling_threshold)
            vertices, triangles = mcubes.marching_cubes(tmpvox, self.sampling_threshold)
            vertices = normalize_vertices(vertices)
            m_smooth = TriangleMesh(vertices, triangles)
            m_smooth.compute_vertex_normals()
            rendered_view = render_geometries([m_smooth], camera_json=CameraJsonPosition, out_img=True)
            rendered_view = Image.fromarray(np.uint8(np.asarray(rendered_view)*255))
            rendered_view = np.asarray(rendered_view.resize((self.real_size, self.real_size)).convert('L'))

            img_x = int(embedded[i, 0])
            img_y = int(embedded[i, 1])
            plt[img_y:img_y + self.real_size, img_x:img_x + self.real_size] = np.minimum(
                plt[img_y:img_y + self.real_size, img_x:img_x + self.real_size], rendered_view)

            img_x = int(embedded[i, 0] / img_size * grid_size)
            img_y = int(embedded[i, 1] / img_size * grid_size)
            if occ_grid[img_y, img_x] == 0:
                img_y = img_y
                img_x = img_x
            elif img_y - 1 >= 0 and occ_grid[img_y - 1, img_x] == 0:
                img_y = img_y - 1
                img_x = img_x
            elif img_y + 1 < grid_size and occ_grid[img_y + 1, img_x] == 0:
                img_y = img_y + 1
                img_x = img_x
            elif img_x - 1 >= 0 and occ_grid[img_y, img_x - 1] == 0:
                img_y = img_y
                img_x = img_x - 1
            elif img_x + 1 < grid_size and occ_grid[img_y, img_x + 1] == 0:
                img_y = img_y
                img_x = img_x + 1
            elif img_y - 1 >= 0 and img_x - 1 >= 0 and occ_grid[img_y - 1, img_x - 1] == 0:
                img_y = img_y - 1
                img_x = img_x - 1
            elif img_y + 1 < grid_size and img_x - 1 >= 0 and occ_grid[img_y + 1, img_x - 1] == 0:
                img_y = img_y + 1
                img_x = img_x - 1
            elif img_y - 1 >= 0 and img_x + 1 < grid_size and occ_grid[img_y - 1, img_x + 1] == 0:
                img_y = img_y - 1
                img_x = img_x + 1
            elif img_y + 1 < grid_size and img_x + 1 < grid_size and occ_grid[img_y + 1, img_x + 1] == 0:
                img_y = img_y + 1
                img_x = img_x + 1
            else:
                print("warning: cannot find spot")
            occ_grid[img_y, img_x] = 1
            img_x *= cell_size
            img_y *= cell_size
            plt_grid[img_y:img_y + self.real_size, img_x:img_x + self.real_size] = np.minimum(
                plt_grid[img_y:img_y + self.real_size, img_x:img_x + self.real_size], rendered_view)

        cv2.imwrite(config.style_codes_dir + "/" + "latent_gz.png", plt)
        cv2.imwrite(config.style_codes_dir + "/" + "latent_gz_grid.png", plt_grid)
        print("rendering...complete")

    def test_fig_3(self, config):

        if not self.load(): exit(-1)

        os.makedirs(config.test_fig_3_dir, exist_ok=True)

        self.imgout_0 = np.full([self.real_size*(3+1), self.real_size*(11+1)], 255, np.uint8)

        for content_idx in range(3):
            # coarse_content_img, _, _ = self.coarse_detailed_full_plots(content_idx)
            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx)
            self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size, 0:self.real_size] = coarse_content_img

        for style_idx in range(3, 14):

            # _, _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx)
            _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx)
            self.imgout_0[0:self.real_size, (style_idx-3 + 1)*self.real_size: (style_idx-3 + 2)*self.real_size] = detailed_full_style_img

            style_data_dict = self.dset.__getitem__(style_idx)
            voxel_style = style_data_dict['voxel_style']
            voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
            z_tensor_g = self.style_encoder(voxel_style, is_training=False)

            for content_idx in range(3):

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
                              (style_idx-3+1)*self.real_size:(style_idx-3+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(config.test_fig_3_dir,f"fig3.png"), self.imgout_0)
