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

from dataset import *
from utils.io_helper import setup_logging
from utils.pytorch3d_vis import CustomDefinedViewMeshRenderer
from utils.matplotlib_utils import render_result, render_example, render_views, plot_matrix

from utils import *
from modelAE_GD import *
import mcubes
from utils.open3d_utils import TriangleMesh
from utils.open3d_render import render_geometries
from utils import CameraJsonPosition
from PIL import Image
import matplotlib.pyplot as plt_original
from utils.nt_xent import original_nt_xent


# cdvr = CustomDefinedViewMeshRenderer(4)
#
#
# def img_of_input(voxel_model_file):
#     with open(voxel_model_file, 'rb') as fin:
#         voxel_model_512 = binvox_rw.read_as_3d_array(fin, fix_coords=True).data.astype(np.uint8)
#     vertices, triangles = mcubes.marching_cubes(voxel_model_512, 0.5)
#     vertices = normalize_vertices(vertices)
#     mcubes_in = cdvr(verts=vertices, triangles=triangles.astype(int))
#     return mcubes_in


class IM_AE(object):

    def coarse_detailed_full_plots(self, i, content=True):
        if content:
            data_dict = self.dset.__getitem__(i)
            tmp, _ = self.dset.get_more(i)
        else:
            data_dict = self.style_set.__getitem__(i)
            tmp, _ = self.style_set.get_more(i)

        xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
        voxel_style = data_dict['voxel_style']

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
            # img_coa2 = self.voxel_renderer.render_img(tmpvox, 0, 6)
            # img_coa = render_views([img_coa1, img_coa2], img_size=self.real_size)
            img_coa = render_views([img_coa1], img_size=self.real_size)
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
            # img_det_full2 = self.voxel_renderer.render_img(tmpvox, 0, 6)
            # img_det_full = render_views([img_det_full1, img_det_full2], img_size=self.real_size)
            img_det_full = render_views([img_det_full1], img_size=self.real_size)
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
    #
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
            # img_smooth_det2 = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, 6)
            # img_smooth_det = render_views([img_smooth_det1, img_smooth_det2], img_size=self.real_size)
            img_smooth_det = render_views([img_smooth_det1], img_size=self.real_size)
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
        self.z_dim = config.style_dim
        self.param_alpha = config.alpha
        self.param_beta = config.beta
        self.nt_xent_factor = config.nt_xent_factor
        self.style_batch = config.style_batch

        self.input_size = config.input_size
        self.output_size = config.output_size

        if self.input_size==64 and self.output_size==256:
            self.upsample_rate = 4
        elif self.input_size==32 and self.output_size==128:
            self.upsample_rate = 4
        elif self.input_size==32 and self.output_size==256:
            self.upsample_rate = 8
        elif self.input_size==16 and self.output_size==128:
            self.upsample_rate = 8
        else:
            print("ERROR: invalid input/output size!")
            exit(-1)

        self.asymmetry = config.asymmetry

        self.save_epoch = 2

        self.sampling_threshold = 0.4

        self.render_view_id = 0
        if self.asymmetry: self.render_view_id = 6 #render side view for motorbike
        self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir

        self.datapath = config.datapath
        self.stylepath = config.stylepath

        self.device = get_torch_device(config, self.log)

        self.bce_loss = nn.BCELoss()


        #load data
        print("preprocessing - start")
        self.log.debug("preprocessing - start")


        assert self.input_size == 16
        assert self.output_size == 128
        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*2], 255, np.uint8)

        self.style_set = BasicDataset(self.data_dir, self.stylepath, config, self.log, filename=config.filename)
        self.styleset_len = len(self.style_set)

        iter_size = min(self.styleset_len, 16)
        if config.train:
            for i in range(iter_size):
                print("preprocessing style - "+str(i+1)+"/"+str(self.styleset_len))

                data_dict = self.style_set.__getitem__(i)
                xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
                voxel_style = data_dict['voxel_style']

                tmp, _ = self.style_set.get_more(i)

                img_y = i//4
                img_x = (i%4)*2+1
                if img_y<4:
                    tmpvox = self.recover_voxel(voxel_style,xmin,xmax,ymin,ymax,zmin,zmax)
                    self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
                img_y = i//4
                img_x = (i%4)*2
                if img_y<4:
                    tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                    tmpvox = self.recover_voxel(tmp_mask_exact,xmin,xmax,ymin,ymax,zmin,zmax)
                    self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)
            
        if config.train: cv2.imwrite(config.sample_dir+"/a_style_0.png", self.imgout_0)



        
        self.imgout_0 = np.full([self.real_size*4, self.real_size*4*2], 255, np.uint8)

        self.dset = BasicDataset(self.data_dir, self.datapath, config, self.log, filename=config.filename)
        self.dataset_len = len(self.dset)

        iter_size = min(self.dataset_len, 16)
        if config.train:
            for i in range(iter_size):
                print("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))
                self.log.debug("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))
                data_dict = self.dset.__getitem__(i)
                xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

                tmp, _ = self.dset.get_more(i)

                img_y = i//4
                img_x = (i%4)*2
                if img_y<4:
                    tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                    tmpvox = self.recover_voxel(tmp_mask_exact,xmin,xmax,ymin,ymax,zmin,zmax)
                    self.imgout_0[img_y*self.real_size:(img_y+1)*self.real_size,img_x*self.real_size:(img_x+1)*self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold, self.render_view_id)


        if config.train: cv2.imwrite(config.sample_dir+"/a_content_0.png", self.imgout_0)
        




        #build model
        self.discriminator = discriminator(self.d_dim,self.styleset_len+1)
        self.discriminator.to(self.device)

        sigmoid = False
        if config.recon_loss == "BCE":
            sigmoid = True
        self.generator = generator_halfsize_x8_allstyles(self.g_dim, self.z_dim, sigmoid)
        self.generator.to(self.device)

        if self.z_dim == 8:
            self.style_encoder = style_encoder_8()
        elif self.z_dim == 16:
            self.style_encoder = style_encoder_16(pool_method=config.pooling, kernel=config.kernel, dilation=config.dilation)
        elif self.z_dim == 32:
            self.style_encoder = style_encoder_32(pool_method=config.pooling, kernel=config.kernel, dilation=config.dilation)
        elif self.z_dim == 64:
            self.style_encoder = style_encoder_64()
        else:
            raise Exception(f"unknown z_dim {self.z_dim}")
        self.style_encoder.to(self.device)

        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr)
        self.optimizer_se = torch.optim.Adam(self.style_encoder.parameters(), lr=config.se_lr)

        #pytorch does not have a checkpoint manager
        #have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 20
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name='IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

    def get_voxel_mask_exact(self,vox):
        #256 -maxpoolk4s4- 64 -upsample- 256
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        #input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = self.upsample_rate, stride = self.upsample_rate, padding = 0)
        #mask
        smallmask_tensor = F.interpolate(smallmaskx_tensor, scale_factor = self.upsample_rate, mode='nearest')
        #to numpy
        smallmask = smallmask_tensor.detach().cpu().numpy()[0,0]
        smallmask = np.round(smallmask).astype(np.uint8)
        return smallmask

    def recover_voxel(self,vox,xmin,xmax,ymin,ymax,zmin,zmax):
        tmpvox = np.zeros([self.real_size,self.real_size,self.real_size], np.float32)
        xmin_,ymin_,zmin_ = (0,0,0)
        xmax_,ymax_,zmax_ = vox.shape
        xmin = xmin*self.upsample_rate-self.mask_margin
        xmax = xmax*self.upsample_rate+self.mask_margin
        ymin = ymin*self.upsample_rate-self.mask_margin
        ymax = ymax*self.upsample_rate+self.mask_margin
        if self.asymmetry:
            zmin = zmin*self.upsample_rate-self.mask_margin
        else:
            zmin = zmin*self.upsample_rate
            zmin_ = self.mask_margin
        zmax = zmax*self.upsample_rate+self.mask_margin
        if xmin<0:
            xmin_ = -xmin
            xmin = 0
        if xmax>self.real_size:
            xmax_ = xmax_+self.real_size-xmax
            xmax = self.real_size
        if ymin<0:
            ymin_ = -ymin
            ymin = 0
        if ymax>self.real_size:
            ymax_ = ymax_+self.real_size-ymax
            ymax = self.real_size
        if zmin<0:
            zmin_ = -zmin
            zmin = 0
        if zmax>self.real_size:
            zmax_ = zmax_+self.real_size-zmax
            zmax = self.real_size
        if self.asymmetry:
            tmpvox[xmin:xmax,ymin:ymax,zmin:zmax] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
        else:
            tmpvox[xmin:xmax,ymin:ymax,zmin:zmax] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
            if zmin*2-zmax-1<0:
                tmpvox[xmin:xmax,ymin:ymax,zmin-1::-1] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
            else:
                tmpvox[xmin:xmax,ymin:ymax,zmin-1:zmin*2-zmax-1:-1] = vox[xmin_:xmax_,ymin_:ymax_,zmin_:zmax_]
        return tmpvox

    def load(self):
        #load previous checkpoint
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

    def save(self,iter):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(iter)+".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
        #delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        #save checkpoint
        torch.save({
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'style_encoder': self.style_encoder.state_dict(),
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

                # use same styles during generator's and discriminator's steps
                style_indices = list(range(self.styleset_len))
                np.random.shuffle(style_indices)

                #ready a fake image
                dxb = batch_index_list[idx]
                content_data_dict = self.dset.__getitem__(dxb)
                mask_content = content_data_dict['mask']
                Dmask_content = content_data_dict['Dmask']
                input_content = content_data_dict['input']

                mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                #D step
                d_step = 1
                for dstep in range(d_step):
                    self.style_encoder.zero_grad()
                    self.discriminator.zero_grad()

                    loss_d_real_total = 0.
                    loss_d_fake_total = 0.
                    all_styles = []
                    all_generated_styles = []
                    for style_idx in style_indices[0:self.style_batch]:
                        style_dict = self.style_set.__getitem__(style_idx)
                        voxel_style = style_dict['voxel_style']
                        voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                        _, Dmask_style = self.style_set.get_more(style_idx)
                        Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

                        D_out = self.discriminator(voxel_style,is_training=True)
                        # loss of "real specific style" + loss of "real any style"
                        loss_d_real = (torch.sum((D_out[:,style_idx:style_idx+1]-1)**2*Dmask_style)
                                      + torch.sum((D_out[:,-1:]-1)**2*Dmask_style))/torch.sum(Dmask_style)
                        loss_d_real = loss_d_real / self.style_batch
                        loss_d_real.backward()  # do backward step to add gradients in correct params
                        loss_d_real_total = loss_d_real_total + loss_d_real.item()

                        z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1, -1, 1, 1, 1])
                        voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)
                        voxel_fake_z_tensor_g = self.style_encoder(voxel_fake, is_training=True)
                        voxel_fake = voxel_fake.detach()  # probably unessesary since correct optimizer is called and
                        # corresponding gradients are set to zero at the start of the corresponding step

                        all_styles.append(z_tensor_g)
                        all_generated_styles.append(voxel_fake_z_tensor_g)

                        D_out = self.discriminator(voxel_fake,is_training=True)
                        # loss of "fake specific style" + loss of "fake any style"
                        loss_d_fake = (torch.sum((D_out[:,style_idx:style_idx+1])**2*Dmask_fake)
                                      + torch.sum((D_out[:,-1:])**2*Dmask_fake))/torch.sum(Dmask_fake)
                        loss_d_fake = loss_d_fake / self.style_batch
                        loss_d_fake.backward()
                        loss_d_fake_total = loss_d_fake_total + loss_d_fake.item()

                    all_generated_styles = torch.cat(all_generated_styles).view(-1, self.z_dim)
                    all_styles = torch.cat(all_styles).view(-1, self.z_dim)

                    loss_d_nt_xent = original_nt_xent(all_generated_styles, all_styles) * self.nt_xent_factor
                    loss_d_nt_xent.backward()  # calculate gradients on correct params

                    self.optimizer_d.step()  # update parameters of D
                    self.optimizer_se.step()  # update parameters of SE

                #recon step
                #reconstruct style image
                r_step = self.style_batch if iter_counter < 5000 else 1  # means after 2 epochs in chairs
                for rstep in range(r_step):
                    qxp = np.random.randint(self.styleset_len)

                    style_data_dict_2 = self.style_set.__getitem__(qxp)
                    voxel_style_2 = style_data_dict_2['voxel_style']
                    mask_style_2 = style_data_dict_2['mask']
                    input_style_2 = style_data_dict_2['input']

                    voxel_style_2 = torch.from_numpy(voxel_style_2).to(self.device).unsqueeze(0).unsqueeze(0)
                    mask_style_2 = torch.from_numpy(mask_style_2).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_style_2 = torch.from_numpy(input_style_2).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    self.style_encoder.zero_grad()
                    self.generator.zero_grad()

                    z_tensor2_g = self.style_encoder(voxel_style_2, is_training=True).view([1,-1,1,1,1])
                    voxel_fake = self.generator(input_style_2,z_tensor2_g,mask_style_2,is_training=True)

                    if config.recon_loss == 'MSE':
                        loss_r = torch.mean((voxel_style_2-voxel_fake)**2)*self.param_beta
                    else:
                        loss_r = self.bce_loss(voxel_fake, voxel_style_2)*self.param_beta
                    loss_r.backward()
                    self.optimizer_g.step()
                    self.optimizer_se.step()
                    del voxel_style_2, mask_style_2, input_style_2  # free up some memory


                # G step
                g_step = 1
                for step in range(g_step):
                    self.style_encoder.zero_grad()
                    self.generator.zero_grad()

                    loss_g_init_total = 0.
                    all_styles = []
                    all_generated_styles = []
                    for style_idx in style_indices[0:self.style_batch]:
                        style_dict = self.style_set.__getitem__(style_idx)
                        voxel_style = style_dict['voxel_style']
                        voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                        z_tensor_g = self.style_encoder(voxel_style, is_training=True)
                        voxel_fake = self.generator(input_fake, z_tensor_g.view([1, -1, 1, 1, 1]), mask_fake,
                                                    is_training=True)
                        voxel_fake_z_tensor_g = self.style_encoder(voxel_fake, is_training=True)
                        D_out = self.discriminator(voxel_fake, is_training=False)
                        loss_g_init = (torch.sum((D_out[:, style_idx:style_idx+1]-1)**2*Dmask_fake)*self.param_alpha
                                      + torch.sum((D_out[:, -1:]-1)**2*Dmask_fake))/torch.sum(Dmask_fake)
                        loss_g_init = loss_g_init / self.style_batch
                        loss_g_init.backward(retain_graph=True)  # calculate gradients on correct params,
                        # but don't delete anything as they will be used in loss_nt_xent.backward
                        loss_g_init_total = loss_g_init_total + loss_g_init.item()

                        all_styles.append(z_tensor_g)
                        all_generated_styles.append(voxel_fake_z_tensor_g)

                    all_generated_styles = torch.cat(all_generated_styles).view(-1, self.z_dim)
                    all_styles = torch.cat(all_styles).view(-1, self.z_dim)

                    loss_g_nt_xent = original_nt_xent(all_generated_styles, all_styles) * self.nt_xent_factor
                    loss_g_nt_xent.backward()  # calculate gradients on correct params

                    self.optimizer_g.step()  # update parameters of G and SE
                    self.optimizer_se.step()  # update parameters of G and SE

                if iter_counter % config.log_iter == 0:
                    print("Iter: [%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_d_nt_xent: %.6f, "
                          "loss_r: %.6f, loss_g_init: %.6f loss_g_nt_xent: %.6f" % (
                          iter_counter, time.time() - start_time,
                          loss_d_real_total, loss_d_fake_total,
                          loss_d_nt_xent.item(),
                          loss_r.item(),
                          loss_g_init_total, loss_g_nt_xent.item()))
                    self.log.debug(
                        "Iter: [%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_d_nt_xent: %.6f, "
                        "loss_r: %.6f, loss_g_init: %.6f loss_g_nt_xent: %.6f" % (
                        iter_counter, time.time() - start_time,
                        loss_d_real_total, loss_d_fake_total,
                        loss_d_nt_xent.item(),
                        loss_r.item(),
                        loss_g_init_total, loss_g_nt_xent.item()))

                    train_writer.add_scalar('loss_d_real', loss_d_real_total, iter_counter)
                    train_writer.add_scalar('loss_d_fake', loss_d_fake_total, iter_counter)
                    train_writer.add_scalar('loss_d_nt_xent', loss_d_nt_xent.item(), iter_counter)
                    train_writer.add_scalar('loss_r', loss_r.item(), iter_counter)
                    train_writer.add_scalar('loss_g_init', loss_g_init_total, iter_counter)
                    train_writer.add_scalar('loss_g_nt_xent', loss_g_nt_xent.item(), iter_counter)

                    heatmap_fig = plot_matrix(all_generated_styles.detach().cpu().numpy(), show=False)
                    train_writer.add_figure('all_generated_styles', heatmap_fig, iter_counter)
                    heatmap_fig = plot_matrix(all_styles.detach().cpu().numpy(), show=False)
                    train_writer.add_figure('all_styles', heatmap_fig, iter_counter)

                    self.visualise(config.sample_dir, f"iter{iter_counter}")

                if iter_counter % config.save_iter == 0:
                    self.save(iter_counter)

            print("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_d_nt_xent: %.6f, "
                  "loss_r: %.6f, loss_g_init: %.6f loss_g_nt_xent: %.6f" % (epoch, training_epoch, time.time() - start_time,
                                                                            loss_d_real_total, loss_d_fake_total,
                                                                            loss_d_nt_xent.item(),
                                                                            loss_r.item(),
                                                                            loss_g_init_total, loss_g_nt_xent.item()))
            self.log.debug("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_d_nt_xent: %.6f, "
                           "loss_r: %.6f, loss_g_init: %.6f loss_g_nt_xent: %.6f" % (epoch, training_epoch, time.time() - start_time,
                                                                            loss_d_real_total, loss_d_fake_total,
                                                                            loss_d_nt_xent.item(),
                                                                            loss_r.item(),
                                                                            loss_g_init_total, loss_g_nt_xent.item()))
            train_writer.add_scalar('loss_d_real', loss_d_real_total, iter_counter)
            train_writer.add_scalar('loss_d_fake', loss_d_fake_total, iter_counter)
            train_writer.add_scalar('loss_d_nt_xent', loss_d_nt_xent.item(), iter_counter)
            train_writer.add_scalar('loss_r', loss_r.item(), iter_counter)
            train_writer.add_scalar('loss_g_init', loss_g_init_total, iter_counter)
            train_writer.add_scalar('loss_g_nt_xent', loss_g_nt_xent.item(), iter_counter)

            heatmap_fig = plot_matrix(all_generated_styles.detach().cpu().numpy(), show=False)
            train_writer.add_figure('all_generated_styles', heatmap_fig, iter_counter)
            heatmap_fig = plot_matrix(all_styles.detach().cpu().numpy(), show=False)
            train_writer.add_figure('all_styles', heatmap_fig, iter_counter)

            self.visualise(config.sample_dir, f"epoch{epoch}")
            self.save(iter_counter)

        #if finish, save
        self.save(iter_counter)

    def visualise_init(self, sample_dir):

        for content_idx in range(5):

            coarse_img, detailed_full_img = self.coarse_detailed_full_plots(content_idx, content=True)

            title = 'Content'

            render_example([coarse_img,
                           detailed_full_img],
                           save=os.path.join(sample_dir, f"content{content_idx}.png"),
                           title=title,
                           titles=['Coarse',  'Detailed Full'])

        for style_idx in range(0, 5):

            coarse_img, detailed_full_img = self.coarse_detailed_full_plots(style_idx, content=False)

            title = 'Style'

            render_example([coarse_img,
                           detailed_full_img],
                           save=os.path.join(sample_dir, f"style{style_idx}.png"),
                           title=title,
                           titles=['Coarse', 'Detailed Full'])


    def visualise(self, sample_dir, epoch):

        self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)

        for style_idx in range(5):
            _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
            self.imgout_0[0:self.real_size, (style_idx+1)*self.real_size:(style_idx+2)*self.real_size] = detailed_full_style_img

        for content_idx in range(5):

            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx, content=True)
            self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size, 0:self.real_size] = coarse_content_img

            content_data_dict = self.dset.__getitem__(content_idx)
            mask_content = content_data_dict['mask']
            input_content = content_data_dict['input']
            xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_idx in range(5):

                style_data_dict = self.style_set.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']

                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                # coarse_img, detailed_img_smooth, detailed_img_full = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                detailed_img_smooth = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)

                self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                              (style_idx+1)*self.real_size: (style_idx+2)*self.real_size] = detailed_img_smooth

        cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)

    def test_style_codes(self, config):

        print("------------------------- PLEASE PROVIDE THE STYLES IN DATAPATH  !!!! -------------------------")
        self.voxel_renderer.use_gpu()

        if self.load() is False: exit(-1)
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

        embedded = TSNE(n_components=2,perplexity=16,learning_rate=10.0,n_iter=2000).fit_transform(style_codes)

        print("rendering...")
        img_size = 5000
        grid_size = 20
        if self.output_size == 128:
            cell_size = 140
        elif self.output_size==256:
            cell_size = 180
        plt = np.full([img_size+self.real_size,img_size+self.real_size],255,np.uint8)
        plt_grid = np.full([grid_size*cell_size+(self.real_size-cell_size),grid_size*cell_size+(self.real_size-cell_size)],255,np.uint8)
        occ_grid = np.zeros([grid_size,grid_size],np.uint8)
        
        x_max = np.max(embedded[:,0])
        x_min = np.min(embedded[:,0])
        y_max = np.max(embedded[:,1])
        y_min = np.min(embedded[:,1])
        x_mid = (x_max+x_min)/2
        y_mid = (y_max+y_min)/2
        scalex = (x_max-x_min)*1.05
        scaley = (y_max-y_min)*1.05
        embedded[:,0] = ((embedded[:,0]-x_mid)/scalex+0.5)*img_size
        embedded[:,1] = ((embedded[:,1]-y_mid)/scaley+0.5)*img_size

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

            img_x = int(embedded[i,0])
            img_y = int(embedded[i,1])
            plt[img_y:img_y+self.real_size,img_x:img_x+self.real_size] = np.minimum(plt[img_y:img_y+self.real_size,img_x:img_x+self.real_size], rendered_view)

            img_x = int(embedded[i,0]/img_size*grid_size)
            img_y = int(embedded[i,1]/img_size*grid_size)
            if occ_grid[img_y,img_x]==0:
                img_y = img_y
                img_x = img_x
            elif img_y-1>=0 and occ_grid[img_y-1,img_x]==0:
                img_y = img_y-1
                img_x = img_x
            elif img_y+1<grid_size and occ_grid[img_y+1,img_x]==0:
                img_y = img_y+1
                img_x = img_x
            elif img_x-1>=0 and occ_grid[img_y,img_x-1]==0:
                img_y = img_y
                img_x = img_x-1
            elif img_x+1<grid_size and occ_grid[img_y,img_x+1]==0:
                img_y = img_y
                img_x = img_x+1
            elif img_y-1>=0 and img_x-1>=0 and occ_grid[img_y-1,img_x-1]==0:
                img_y = img_y-1
                img_x = img_x-1
            elif img_y+1<grid_size and img_x-1>=0 and occ_grid[img_y+1,img_x-1]==0:
                img_y = img_y+1
                img_x = img_x-1
            elif img_y-1>=0 and img_x+1<grid_size and occ_grid[img_y-1,img_x+1]==0:
                img_y = img_y-1
                img_x = img_x+1
            elif img_y+1<grid_size and img_x+1<grid_size and occ_grid[img_y+1,img_x+1]==0:
                img_y = img_y+1
                img_x = img_x+1
            else:
                print("warning: cannot find spot")
            occ_grid[img_y,img_x]=1
            img_x *= cell_size
            img_y *= cell_size
            plt_grid[img_y:img_y + self.real_size, img_x:img_x + self.real_size] = np.minimum(
                plt_grid[img_y:img_y + self.real_size, img_x:img_x + self.real_size], rendered_view)

        cv2.imwrite(config.style_codes_dir + "/" + "latent_gz.png", plt)
        cv2.imwrite(config.style_codes_dir + "/" + "latent_gz_grid.png", plt_grid)
        print("rendering...complete")
    def test_fig_3(self, config):

        style_indices = [6, 17, 21, 22, 25, 27, 3, 15, 20]

        self.voxel_renderer.use_gpu()

        if self.load() is False: exit(-1)

        os.makedirs(config.test_fig_3_dir, exist_ok=True)

        self.imgout_0 = np.full([self.real_size*(self.dataset_len+1), self.real_size*(len(style_indices)+1)], 255, np.uint8)

        for content_idx in range(self.dataset_len):
            # coarse_content_img, _, _ = self.coarse_detailed_full_plots(content_idx, content=True)
            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx, content=True)
            self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size, 0:self.real_size] = coarse_content_img

        for style_idx in range(self.styleset_len):
            if style_idx not in style_indices:
                continue

            # _, _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
            _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
            self.imgout_0[0:self.real_size, (style_indices.index(style_idx) + 1)*self.real_size: (style_indices.index(style_idx) + 2)*self.real_size] = detailed_full_style_img

            voxel_style = self.style_set.__getitem__(style_idx)['voxel_style']
            voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
            z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1, -1, 1, 1, 1])

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


    def prepare_voxel_style(self, config):
        import sys
        sys.path.extend(os.path.dirname(os.path.abspath(__file__)))
        from common_utils import binvox_rw_faster as binvox_rw
        #import mcubes

        result_dir = "output_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        max_num_of_styles = 16
        max_num_of_contents = 20


        #load style shapes
        fin = open("splits/"+self.data_style+".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        for style_id in range(self.styleset_len):
            print("preprocessing style - "+str(style_id+1)+"/"+str(self.styleset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.styleset_names[style_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.styleset_names[style_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)

            #tmp = gaussian_filter(tmp.astype(np.float32), sigma=1)
            #tmp = (tmp>self.sampling_threshold).astype(np.uint8)

            binvox_rw.write_voxel(tmp, result_dir+"/style_"+str(style_id)+".binvox")
            # tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            # binvox_rw.write_voxel(tmp_input, result_dir+"/style_"+str(style_id)+"_coarse.binvox")

            # vertices, triangles = mcubes.marching_cubes(tmp, 0.5)
            # vertices = vertices-0.5
            # write_ply_triangle(result_dir+"/style_"+str(style_id)+".ply", vertices, triangles)
            # vertices, triangles = mcubes.marching_cubes(tmp_input, 0.5)
            # vertices = (vertices-0.5)*4.0
            # write_ply_triangle(result_dir+"/style_"+str(style_id)+"_coarse.ply", vertices, triangles)



    def prepare_voxel_for_eval(self, config):
        import sys
        sys.path.extend(os.path.dirname(os.path.abspath(__file__)))
        from common_utils import binvox_rw_faster as binvox_rw
        #import mcubes

        result_dir = "output_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if self.load() is False: exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 20


        #load style shapes
        fin = open("splits/"+self.data_style+".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        #load content shapes
        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - "+str(content_id+1)+"/"+str(self.dataset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
            
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            binvox_rw.write_voxel(tmp_input, result_dir+"/content_"+str(content_id)+"_coarse.binvox")

            # vertices, triangles = mcubes.marching_cubes(tmp_input, 0.5)
            # vertices = (vertices-0.5)*4.0
            # write_ply_triangle(result_dir+"/content_"+str(content_id)+"_coarse.ply", vertices, triangles)

            mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()


            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len],np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]
                tmp_voxel_fake = (tmp_voxel_fake>self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmp_voxel_fake, result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox")

                # vertices, triangles = mcubes.marching_cubes(tmp_voxel_fake, 0.5)
                # vertices = vertices-0.5
                # write_ply_triangle(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".ply", vertices, triangles)



    def prepare_voxel_for_FID(self, config):
        import sys
        sys.path.extend(os.path.dirname(os.path.abspath(__file__)))
        from common_utils import binvox_rw_faster as binvox_rw
        #import mcubes

        result_dir = "output_for_FID"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if self.load() is False: exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 100


        #load style shapes
        fin = open("splits/"+self.data_style+".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        #load content shapes
        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - "+str(content_id+1)+"/"+str(self.dataset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            
            mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len],np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]
                tmpvox = self.recover_voxel(tmp_voxel_fake,xmin,xmax,ymin,ymax,zmin,zmax)
                tmpvox = (tmpvox>self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmpvox, result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox")




    def render_fake_for_eval(self, config):

        self.voxel_renderer.use_gpu()

        result_dir = "render_fake_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if self.load() is False: exit(-1)

        sample_num_views = 24
        render_boundary_padding_size = 16
        half_real_size = self.real_size//2
        max_num_of_styles = 16
        max_num_of_contents = 100


        #load style shapes
        fin = open("splits/"+self.data_style+".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)


        #load content shapes
        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - "+str(content_id+1)+"/"+str(self.dataset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)
            
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
            mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()


            tmpvoxlarger = np.zeros([self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2], np.float32)
            
            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len],np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]

                xmin2 = xmin*self.upsample_rate-self.mask_margin
                xmax2 = xmax*self.upsample_rate+self.mask_margin
                ymin2 = ymin*self.upsample_rate-self.mask_margin
                ymax2 = ymax*self.upsample_rate+self.mask_margin
                if self.asymmetry:
                    zmin2 = zmin*self.upsample_rate-self.mask_margin
                else:
                    zmin2 = zmin*self.upsample_rate
                zmax2 = zmax*self.upsample_rate+self.mask_margin

                if self.asymmetry:
                    tmpvoxlarger[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,:]
                else:
                    tmpvoxlarger[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]
                    tmpvoxlarger[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2-1+render_boundary_padding_size:zmin2*2-zmax2-1+render_boundary_padding_size:-1] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]

                for sample_id in range(sample_num_views):
                    cam_alpha = np.random.random()*np.pi*2
                    cam_beta = np.random.random()*np.pi/2-np.pi/4
                    tmpvoxlarger_tensor = torch.from_numpy(tmpvoxlarger).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    imgout = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvoxlarger_tensor, self.sampling_threshold, cam_alpha, cam_beta, get_depth = False, processed = True)
                    if self.output_size==128:
                        imgout = cv2.resize(imgout,(self.real_size*2,self.real_size*2), interpolation=cv2.INTER_NEAREST)
                        imgout = imgout[half_real_size:-half_real_size,half_real_size:-half_real_size]
                    cv2.imwrite(result_dir+"/"+str(content_id)+"_"+str(style_id)+"_"+str(sample_id)+".png", imgout)





    def render_real_for_eval(self, config):

        self.voxel_renderer.use_gpu()

        result_dir = "render_real_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        sample_num_views = 24
        render_boundary_padding_size = 16
        half_real_size = self.real_size//2


        #load all shapes
        fin = open("splits/"+self.data_content+".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)


        for content_id in range(self.dataset_len):
            print("processing content - "+str(content_id+1)+"/"+str(self.dataset_len))
            if self.output_size==128:
                tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
            elif self.output_size==256:
                tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)

            #tmp_raw = gaussian_filter(tmp_raw.astype(np.float32), sigma=1)

            for sample_id in range(sample_num_views):
                cam_alpha = np.random.random()*np.pi*2
                cam_beta = np.random.random()*np.pi/2-np.pi/4
                imgout = self.voxel_renderer.render_img_with_camera_pose_gpu(tmp_raw, self.sampling_threshold, cam_alpha, cam_beta, get_depth = False, processed = False)
                if self.output_size==128:
                    imgout = cv2.resize(imgout,(self.real_size*2,self.real_size*2), interpolation=cv2.INTER_NEAREST)
                    imgout = imgout[half_real_size:-half_real_size,half_real_size:-half_real_size]
                cv2.imwrite(result_dir+"/"+str(content_id)+"_"+str(sample_id)+".png", imgout)





    def launch_ui(self, config):
        from scipy.spatial import Delaunay

        use_precomputed_tsne = False
        self.sampling_threshold = 0.25

        #Ubuntu python did not come with Tkinter and I was too lazy to install it.
        #Therefore the entire UI is just a huge image.
        UI_imgheight = 800
        UI_height = 1000
        UI_width = UI_imgheight+UI_height
        UI_image_ = np.full([UI_height,UI_width,3], 255, np.uint8)

        self.voxel_renderer.use_gpu()

        if self.load() is False: exit(-1)

        style_codes = self.generator.style_codes.detach().cpu().numpy()
        style_codes = (style_codes-np.mean(style_codes,axis=0))/np.std(style_codes,axis=0)


        if not use_precomputed_tsne:
            #compute
            embedded = TSNE(n_components=2,perplexity=16,learning_rate=10.0,n_iter=2000).fit_transform(style_codes)
            fout = open(config.sample_dir+"/"+"tsne_coords.txt", 'w')
            for i in range(self.styleset_len):
                fout.write( str(embedded[i,0])+"\t"+str(embedded[i,1])+"\n" )
            fout.close()
        else:
            #load computed
            embedded = np.zeros([self.styleset_len,2], np.float32)
            fin = open(config.sample_dir+"/"+"tsne_coords.txt")
            lines = fin.readlines()
            fin.close()
            for i in range(self.styleset_len):
                line = lines[i].split()
                embedded[i,0] = float(line[0])
                embedded[i,1] = float(line[1])




        if self.output_size==128:
            img_size = 2048
        elif self.output_size==256:
            img_size = 4096
        x_max = np.max(embedded[:,0])
        x_min = np.min(embedded[:,0])
        y_max = np.max(embedded[:,1])
        y_min = np.min(embedded[:,1])
        x_mid = (x_max+x_min)/2
        y_mid = (y_max+y_min)/2
        scalex = (x_max-x_min)*1.0
        scaley = (y_max-y_min)*1.0
        embedded[:,0] = ((embedded[:,0]-x_mid)/scalex+0.5)*img_size
        embedded[:,1] = ((embedded[:,1]-y_mid)/scaley+0.5)*img_size


        if not use_precomputed_tsne:
            #render
            print("rendering...")
            plt = np.full([img_size+self.real_size,img_size+self.real_size],255,np.uint8)
            for i in range(self.styleset_len):
                if self.output_size==128:
                    tmpvox = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.styleset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
                elif self.output_size==256:
                    tmpvox = get_vox_from_binvox(os.path.join(self.data_dir,self.styleset_names[i]+"/model_depth_fusion.binvox")).astype(np.uint8)
                rendered_view = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvox, self.sampling_threshold)
                img_x = int(embedded[i,0])
                img_y = int(embedded[i,1])
                plt[img_y:img_y+self.real_size,img_x:img_x+self.real_size] = np.minimum(plt[img_y:img_y+self.real_size,img_x:img_x+self.real_size], rendered_view)
            cv2.imwrite(config.sample_dir+"/"+"latent_gz.png", plt)
            print("rendering...complete")
        else:
            #load rendered
            plt = cv2.imread(config.sample_dir+"/"+"latent_gz.png", cv2.IMREAD_UNCHANGED)




        #rescale embedding image
        rescale_factor = UI_height/(img_size+self.real_size)
        plt = cv2.resize(plt, (UI_height,UI_height))
        plt[0,:] = 205
        plt[-1,:] = 205
        plt[:,0] = 205
        plt[:,-1] = 205
        plt = np.reshape(plt,[UI_height,UI_height,1])
        UI_image_[:,UI_imgheight:] = plt

        render_boundary_padding_size = 16

        content_offset_x = 100
        content_offset_y = UI_imgheight + 20
        content_spacing = 20
        content_textlen = 400
        content_max_len = 8
        content_start = 0
        content_id = 0
        content_id_changed_flag = True
        font = cv2.FONT_HERSHEY_SIMPLEX

        scrollbar_offset_x = content_offset_x - 30
        scrollbar_offset_x2 = content_offset_x - 10
        scrollbar_offset_y = content_offset_y - content_spacing
        scrollbar_offset_y2 = scrollbar_offset_y + content_max_len*content_spacing
        scrollbar_height = content_max_len*content_spacing

        content_img_size = 200
        content_img_offset_x = 550
        content_img_offset_y = UI_imgheight
        quater_real_size = self.real_size//4
        half_real_size = self.real_size//2

        cam_alpha = 0.785
        cam_beta = 0.785


        embedded[:,0] = (embedded[:,0] +half_real_size)*rescale_factor
        embedded[:,1] = (embedded[:,1] +half_real_size)*rescale_factor
        tsne_x = int(embedded[0,0])
        tsne_y = int(embedded[0,1])
        z_vector = np.zeros([self.styleset_len],np.float32)
        z_vector[0] = 1
        z_vector_changed_flag = True


        #prepare the triangulation and barycentric coordinates
        #https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        tri = Delaunay(embedded)
        tri_index = tri.simplices
        points_idxs = np.linspace(0,UI_height-1, UI_height, dtype = np.float32)
        points_x, points_y = np.meshgrid(points_idxs,points_idxs, sparse=False, indexing='ij')
        points_x = np.reshape(points_x, [UI_height*UI_height,1])
        points_y = np.reshape(points_y, [UI_height*UI_height,1])
        points = np.concatenate([points_y,points_x], 1)
        row_idx = tri.find_simplex(points)
        X = tri.transform[row_idx, :2]
        Y = points - tri.transform[row_idx, 2]
        b = np.einsum('...jk,...k->...j', X, Y)
        bcoords = np.c_[b, 1-b.sum(axis=1)]
        bcoords = np.reshape(bcoords, [UI_height,UI_height,3])
        valid_mask = np.reshape(row_idx>=0, [UI_height,UI_height,1]).astype(np.uint8)
        row_idx = np.reshape(row_idx, [UI_height,UI_height])
        UI_image_[:,UI_imgheight:] = np.minimum(UI_image_[:,UI_imgheight:], valid_mask*15+240)


        #capture mouse events
        mouse_xyd = np.zeros([3], np.int32)
        mouse_xyd_backup = np.zeros([3], np.int32)
        def mouse_ops(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_xyd[2] = 1
                mouse_xyd[0] = x
                mouse_xyd[1] = y
            elif event == cv2.EVENT_MOUSEMOVE:
                if mouse_xyd[2] == 1:
                    mouse_xyd[0] = x
                    mouse_xyd[1] = y
            elif event == cv2.EVENT_LBUTTONUP:
                mouse_xyd[2] = 0


        Window_name = "Explorer"
        cv2.namedWindow(Window_name)
        cv2.setMouseCallback(Window_name,mouse_ops)

        #UI starts
        while True:

            #deal with mouse events
            if mouse_xyd[0]!=mouse_xyd_backup[0] or mouse_xyd[1]!=mouse_xyd_backup[1] or mouse_xyd[2]!=mouse_xyd_backup[2]:

                if mouse_xyd[0]<UI_imgheight and mouse_xyd[1]<UI_imgheight: #inside output rergion
                    if mouse_xyd_backup[0]<UI_imgheight and mouse_xyd_backup[1]<UI_imgheight and mouse_xyd[2]==1 and mouse_xyd_backup[2]==1:
                        dx = mouse_xyd[0] - mouse_xyd_backup[0]
                        dy = mouse_xyd[1] - mouse_xyd_backup[1]
                        cam_alpha += dx/200.0
                        cam_beta += dy/200.0
                        if cam_beta>1.2: cam_beta=1.2
                        if cam_beta<-1.2: cam_beta=-1.2

                elif mouse_xyd[0]>UI_imgheight: #inside tsne rergion
                    if mouse_xyd[2]==1:
                        this_row_idx = row_idx[mouse_xyd[1],mouse_xyd[0]-UI_imgheight]
                        if this_row_idx>=0:
                            tsne_x = mouse_xyd[0]-UI_imgheight
                            tsne_y = mouse_xyd[1]
                            this_tri_index = tri_index[this_row_idx]
                            this_bcoords = bcoords[tsne_y,tsne_x]
                            z_vector[:] = 0
                            for i in range(3):
                                z_vector[this_tri_index[i]] = this_bcoords[i]
                            z_vector_changed_flag = True

                elif mouse_xyd[0]>=scrollbar_offset_x and mouse_xyd[0]<scrollbar_offset_x2 and mouse_xyd[1]>=scrollbar_offset_y and mouse_xyd[1]<scrollbar_offset_y2+20: #inside scrollbar
                    if mouse_xyd[2]==1:
                        dy = float(mouse_xyd[1] -10 - scrollbar_offset_y)/scrollbar_height
                        content_start = int(self.dataset_len*dy)
                        if content_start<0: content_start=0
                        if content_start>=self.dataset_len-content_max_len: content_start=self.dataset_len-content_max_len-1

                elif mouse_xyd[0]>=content_offset_x and mouse_xyd[0]<content_offset_x + content_textlen and mouse_xyd[1]>=scrollbar_offset_y and mouse_xyd[1]<scrollbar_offset_y2: #inside content shape browser
                    if mouse_xyd[2]==1 and mouse_xyd_backup[2]==0:
                        dy = mouse_xyd[1] - scrollbar_offset_y
                        content_id = content_start + dy//content_spacing
                        content_id_changed_flag = True
                        z_vector_changed_flag = True

                mouse_xyd_backup[:] = mouse_xyd[:]

            #put embedding image
            UI_image = np.copy(UI_image_)
            text_x = tsne_x  +UI_imgheight -5
            text_x2 = text_x +10
            text_x2 = min(text_x2, UI_width)
            text_y = tsne_y -5
            text_y2 = text_y +10
            text_y = max(text_y, 0)
            text_y2 = min(text_y2, UI_height)
            UI_image[text_y:text_y2,text_x:text_x2, 1:3] = 0

            #put content shape browser
            text_x = content_offset_x
            text_x2 = content_offset_x + content_textlen
            text_y = scrollbar_offset_y
            text_y2 = scrollbar_offset_y2
            UI_image[text_y:text_y2,text_x:text_x2] = 240

            #scrollbar
            text_x = scrollbar_offset_x
            text_x2 = scrollbar_offset_x2
            text_y = scrollbar_offset_y
            text_y2 = scrollbar_offset_y2
            UI_image[text_y:text_y2,text_x:text_x2] = 240

            dy = float(mouse_xyd[1] - scrollbar_offset_y)/scrollbar_height
            content_start//self.dataset_len * scrollbar_height

            content_id_pos = content_start*scrollbar_height//self.dataset_len
            text_x = scrollbar_offset_x
            text_x2 = scrollbar_offset_x2
            text_y = scrollbar_offset_y + content_id_pos
            text_y2 = text_y + 20
            UI_image[text_y:text_y2,text_x:text_x2] = 205

            #highlight
            relative_pos = content_id - content_start
            if relative_pos>=0 and relative_pos<content_max_len:
                text_x = content_offset_x
                text_x2 = content_offset_x + content_textlen
                text_y = scrollbar_offset_y + relative_pos*content_spacing
                text_y2 = text_y + content_spacing
                UI_image[text_y:text_y2,text_x:text_x2,0] = 255
                UI_image[text_y:text_y2,text_x:text_x2,1] = 232
                UI_image[text_y:text_y2,text_x:text_x2,2] = 204
            
            #texts of names
            for i in range(content_max_len):
                text = self.dataset_names[i+content_start]
                text_x = content_offset_x
                text_y = content_offset_y + i*content_spacing -5
                cv2.putText(UI_image, text, (text_x, text_y), font, 0.5, (0,0,0), 1)

            #loading content shape
            if content_id_changed_flag:
                content_id_changed_flag = False

                if self.output_size==128:
                    tmp_raw = get_vox_from_binvox_1over2(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
                elif self.output_size==256:
                    tmp_raw = get_vox_from_binvox(os.path.join(self.data_dir,self.dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
                xmin,xmax,ymin,ymax,zmin,zmax = self.get_voxel_bbox(tmp_raw)
                tmp = self.crop_voxel(tmp_raw,xmin,xmax,ymin,ymax,zmin,zmax)

                tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)
                mask_fake  = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

                tmp_voxel_fake = self.get_voxel_mask_exact(tmp)

                contentvox = np.zeros([self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2], np.float32)

                xmin2 = xmin*self.upsample_rate-self.mask_margin
                xmax2 = xmax*self.upsample_rate+self.mask_margin
                ymin2 = ymin*self.upsample_rate-self.mask_margin
                ymax2 = ymax*self.upsample_rate+self.mask_margin
                if self.asymmetry:
                    zmin2 = zmin*self.upsample_rate-self.mask_margin
                else:
                    zmin2 = zmin*self.upsample_rate
                zmax2 = zmax*self.upsample_rate+self.mask_margin

                if self.asymmetry:
                    contentvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,:]
                else:
                    contentvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]
                    contentvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2-1+render_boundary_padding_size:zmin2*2-zmax2-1+render_boundary_padding_size:-1] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]

                contentvox_tensor = torch.from_numpy(contentvox).to(self.device).unsqueeze(0).unsqueeze(0).float()

            img = self.voxel_renderer.render_img_with_camera_pose_gpu(contentvox_tensor, self.sampling_threshold, cam_alpha, cam_beta, get_depth = False, processed = True)
            img = cv2.resize(img, (content_img_size,content_img_size))
            UI_image[content_img_offset_y:content_img_offset_y+content_img_size,content_img_offset_x:content_img_offset_x+content_img_size] = np.reshape(img,[content_img_size,content_img_size,1])



            #running the network
            if z_vector_changed_flag:
                z_vector_changed_flag = False

                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])
                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0,0]

                outputvox = np.zeros([self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2,self.real_size+render_boundary_padding_size*2], np.float32)

                xmin2 = xmin*self.upsample_rate-self.mask_margin
                xmax2 = xmax*self.upsample_rate+self.mask_margin
                ymin2 = ymin*self.upsample_rate-self.mask_margin
                ymax2 = ymax*self.upsample_rate+self.mask_margin
                if self.asymmetry:
                    zmin2 = zmin*self.upsample_rate-self.mask_margin
                else:
                    zmin2 = zmin*self.upsample_rate
                zmax2 = zmax*self.upsample_rate+self.mask_margin

                if self.asymmetry:
                    outputvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,:]
                else:
                    outputvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2+render_boundary_padding_size:zmax2+render_boundary_padding_size] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]
                    outputvox[xmin2+render_boundary_padding_size:xmax2+render_boundary_padding_size,ymin2+render_boundary_padding_size:ymax2+render_boundary_padding_size,zmin2-1+render_boundary_padding_size:zmin2*2-zmax2-1+render_boundary_padding_size:-1] = tmp_voxel_fake[::-1,::-1,self.mask_margin:]

                outputvox_tensor = torch.from_numpy(outputvox).to(self.device).unsqueeze(0).unsqueeze(0).float()

            img = self.voxel_renderer.render_img_with_camera_pose_gpu(outputvox_tensor, self.sampling_threshold, cam_alpha, cam_beta, get_depth = False, processed = True)
            img = cv2.resize(img, (UI_imgheight,UI_imgheight))
            
            UI_image[:UI_imgheight,:UI_imgheight] = np.reshape(img,[UI_imgheight,UI_imgheight,1])


            cv2.imshow(Window_name, UI_image)
            key = cv2.waitKey(1)
            if key == 32: #space
                break


