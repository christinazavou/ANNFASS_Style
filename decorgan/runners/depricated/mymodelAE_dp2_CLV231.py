import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataset2 import *
from runners.common_dp2 import IM_AE_MYMODEL_COMMON
from utils.io_helper import setup_logging
from utils.pytorch3d_vis import CustomDefinedViewMeshRenderer

from utils import *
from modelAE_GD import *
import mcubes
from utils.nt_xent import original_nt_xent
from utils.matplotlib_utils import plot_matrix


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

        self.g_dim = 32
        self.d_dim = 32
        self.z_dim = config.style_dim
        self.s_dim = 16
        self.param_alpha = config.alpha
        self.param_beta = config.beta
        self.nt_xent_factor = config.nt_xent_factor
        self.style_batch = config.style_batch
        self.tau = config.tau

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

        iter_size = self.dataset_len
        if config.debug:
            iter_size = 10

        if config.train:
            for i in range(iter_size):
                print("preprocessing style - "+str(i+1)+"/"+str(self.dataset_len))
                self.log.debug("preprocessing style - "+str(i+1)+"/"+str(self.dataset_len))
                data_dict = self.dset.__getitem__(i)
                xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
                voxel_style = data_dict['voxel_style']

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
                    tmp, _ = self.dset.get_more(i)
                    tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                    tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                    self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                    img_x * self.real_size:(img_x + 1) * self.real_size] = self.voxel_renderer.render_img(tmpvox,
                                                                                                          self.sampling_threshold,
                                                                                                          self.render_view_id)


        if config.train: cv2.imwrite(config.sample_dir + "/a_style_0.png", self.imgout_0)

        self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)

        if config.train:
            for i in range(iter_size):
                print("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))
                self.log.debug("preprocessing content - "+str(i+1)+"/"+str(self.dataset_len))
                data_dict = self.dset.__getitem__(i)
                xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

                img_y = i // 4
                img_x = (i % 4) * 2
                if img_y < 4:
                    tmp, _ = self.dset.get_more(i)
                    tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                    tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                    self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                    img_x * self.real_size:(img_x + 1) * self.real_size] = self.voxel_renderer.render_img(tmpvox,
                                                                                                          self.sampling_threshold,
                                                                                                          self.render_view_id)

        if config.train: cv2.imwrite(config.sample_dir + "/a_content_0.png", self.imgout_0)

        # build model
        self.discriminator = discriminator(self.d_dim, 1)
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

                content_data_dict = self.dset.__getitem__(content_idx)
                input_content = content_data_dict['input']
                Dmask_content = content_data_dict['Dmask']
                mask_content = content_data_dict['mask']

                mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                Dmask_fake = torch.from_numpy(Dmask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                # use same styles during generator's and discriminator's steps
                style_indices = list(set(range(self.dataset_len)) - {content_idx})
                np.random.shuffle(style_indices)

                # D step
                d_step = 1
                for dstep in range(d_step):
                    # removed style_encoder update !!
                    self.discriminator.zero_grad()

                    loss_d_real_total = 0.
                    loss_d_fake_total = 0.
                    all_styles = []
                    all_generated_styles = []
                    for style_idx in style_indices[0:self.style_batch]:
                        style_dict = self.dset.__getitem__(style_idx)
                        voxel_style = style_dict['voxel_style']
                        Dmask_style = style_dict['Dmask_style']
                        voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                        Dmask_style = torch.from_numpy(Dmask_style).to(self.device).unsqueeze(0).unsqueeze(0).float()

                        D_out = self.discriminator(voxel_style, is_training=True)
                        loss_d_real = torch.sum((D_out - 1) ** 2 * Dmask_style) / torch.sum(Dmask_style)
                        loss_d_real = loss_d_real / self.style_batch
                        loss_d_real.backward()
                        loss_d_real_total = loss_d_real_total + loss_d_real.item()

                        z_tensor_g = self.style_encoder(voxel_style, is_training=False)
                        voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)
                        voxel_fake_z_tensor_g = self.style_encoder(voxel_fake, is_training=True)
                        voxel_fake = voxel_fake.detach()  # probably unessesary since correct optimizer is called and
                        # corresponding gradients are set to zero at the start of the corresponding step

                        all_styles.append(z_tensor_g)
                        all_generated_styles.append(voxel_fake_z_tensor_g)

                        D_out = self.discriminator(voxel_fake, is_training=True)
                        loss_d_fake = torch.sum(D_out ** 2 * Dmask_fake) / torch.sum(Dmask_fake)
                        loss_d_fake = loss_d_fake / self.style_batch
                        loss_d_fake.backward()
                        loss_d_fake_total = loss_d_fake_total + loss_d_fake.item()

                    all_generated_styles = torch.cat(all_generated_styles).view(-1, self.z_dim)
                    all_styles = torch.cat(all_styles).view(-1, self.z_dim)

                    loss_d_nt_xent = original_nt_xent(all_generated_styles, all_styles, tau=self.tau) * self.nt_xent_factor
                    loss_d_nt_xent.backward()  # calculate gradients on correct params

                    self.optimizer_d.step()  # update parameters of D

                # recon step
                # reconstruct style image
                r_step = self.style_batch if iter_counter < 5000 else 1  # means after 2 epochs in chairs
                for rstep in range(r_step):
                    style_idx_2 = np.random.randint(self.dataset_len)

                    style_data_dict_2 = self.dset.__getitem__(style_idx_2)
                    input_style_2 = style_data_dict_2['input']
                    mask_style_2 = style_data_dict_2['mask']
                    voxel_style_2 = style_data_dict_2['voxel_style']

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
                        voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=True)
                        voxel_fake_z_tensor_g = self.style_encoder(voxel_fake, is_training=True)
                        D_out = self.discriminator(voxel_fake, is_training=False)
                        loss_g_init = torch.sum((D_out-1)**2*Dmask_fake)*self.param_alpha/torch.sum(Dmask_fake)
                        loss_g_init = loss_g_init / self.style_batch
                        loss_g_init.backward(retain_graph=True)  # calculate gradients on correct params,
                        # but don't delete anything as they will be used in loss_nt_xent.backward
                        loss_g_init_total = loss_g_init_total + loss_g_init.item()

                        all_styles.append(z_tensor_g)
                        all_generated_styles.append(voxel_fake_z_tensor_g)

                    all_generated_styles = torch.cat(all_generated_styles).view(-1, self.z_dim)
                    all_styles = torch.cat(all_styles).view(-1, self.z_dim)

                    loss_g_nt_xent = original_nt_xent(all_generated_styles, all_styles, tau=self.tau) * self.nt_xent_factor
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

            epoch += 1
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

        # if finish, save
        self.save(iter_counter)
