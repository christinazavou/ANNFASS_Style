from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.matplotlib_utils import render_result, render_example, render_views, plot_matrix

from utils import *
from common_utils.utils import normalize_vertices
from modelAE_GD import *
import mcubes
from utils.open3d_utils import TriangleMesh, PointCloud
from utils.open3d_render import render_geometries
from utils import CameraJsonPosition
from PIL import Image
from utils.triplet_loader import get_loader
from tensorboardX import SummaryWriter
from utils.io_helper import setup_logging
import git


def plot_grad_flow(named_parameters, title=None):
    # todo: maybe use exp(x) or -1/log(x) to make the gradients appear larger on plots
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    title = f"Gradient flow" if not title else f"Gradient flow {title}"

    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)

    ax_full = fig.add_subplot(gs[0, 0])
    ax_focus = fig.add_subplot(gs[0, 1])

    ax_full.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax_full.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax_full.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    ax_full.set_xticks(range(0, len(ave_grads), 1))
    ax_full.set_xticklabels(layers, rotation=60)
    ax_full.set_xlim(left=0, right=len(ave_grads))
    ax_full.set_xlabel("Layers")
    ax_full.set_ylabel("average gradient")
    ax_full.grid(True)
    ax_full.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    ax_focus.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax_focus.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax_focus.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    ax_focus.set_xticks(range(0, len(ave_grads), 1))
    ax_focus.set_xticklabels(layers, rotation=60)
    ax_focus.set_xlim(left=0, right=len(ave_grads))
    ax_focus.set_ylim(bottom=-0.02, top=0.02)  # zoom in on the lower gradient regions
    ax_focus.set_xlabel("Layers")
    ax_focus.set_ylabel("average gradient")
    ax_focus.grid(True)
    ax_focus.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.suptitle(title)
    plt.tight_layout()
    return fig


class IM_AE_STATIC(object):

    @staticmethod
    def vox_plot(vox, local, voxel_renderer, sampling_threshold, real_size):
        if not local:
            img_smooth_det1 = voxel_renderer.render_img(vox, sampling_threshold, 0)
            img_smooth_det2 = voxel_renderer.render_img(vox, sampling_threshold, 6)
            img_smooth_det = render_views([img_smooth_det1, img_smooth_det2], img_size=real_size)
        else:
            # vertices = get_points_from_voxel(vox, sampling_threshold)
            vertices, triangles = mcubes.marching_cubes(vox, sampling_threshold)
            if len(vertices) == 0:
                img_smooth_det = np.ones((real_size, real_size)).astype(np.uint8) * 255
            else:
                vertices = normalize_vertices(vertices)
                m_smooth = TriangleMesh(vertices, triangles)
                m_smooth.compute_vertex_normals()
                # m_smooth = PointCloud(vertices)
                img_smooth_det = render_geometries([m_smooth], camera_json=CameraJsonPosition, out_img=True)
                img_smooth_det = Image.fromarray(np.uint8(np.asarray(img_smooth_det) * 255))
                img_smooth_det = np.asarray(img_smooth_det.resize((real_size, real_size)).convert('L'))
        return img_smooth_det

    @staticmethod
    def get_voxel_mask_exact(vox,
                             device, upsample_rate):
        #256 -maxpoolk4s4- 64 -upsample- 256
        vox_tensor = torch.from_numpy(vox).to(device).unsqueeze(0).unsqueeze(0).float()
        #input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=upsample_rate, stride=upsample_rate, padding=0)
        #mask
        smallmask_tensor = F.interpolate(smallmaskx_tensor, scale_factor=upsample_rate, mode='nearest')
        #to numpy
        smallmask = smallmask_tensor.detach().cpu().numpy()[0, 0]
        smallmask = np.round(smallmask).astype(np.uint8)
        return smallmask


class IM_AE_COMMON(object):

    def _init_common_config(self, config):
        if os.path.exists("/home/graphicslab/"):
            self.local = True
        else:
            self.local = False

        log_dir = os.path.dirname(config.checkpoint_dir)
        if config.finetune:
            setup_logging(log_dir, f"log_finetune_{config.pct_suffix}.txt")
        else:
            setup_logging(log_dir)
        self.log = logging.getLogger(self.__class__.__name__)

        print(f"initializing {self.__class__.__name__}...")
        self.log.debug(f"initializing {self.__class__.__name__}...")
        print(f"config:\n{config}")
        self.log.debug(f"config:\n{config}")

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f"gir repo sha:\n{sha}")
        self.log.debug(f"git repo sha:\n{sha}")

        self.real_size = 256
        self.mask_margin = 8

        self.g_dim = config.gen_dim
        self.d_dim = config.disc_dim

        self.param_alpha = config.alpha
        self.param_beta = config.beta

        self.g_steps = config.g_steps
        self.d_steps = config.d_steps
        self.r_steps = config.r_steps

        self.input_size = config.input_size
        self.output_size = config.output_size

        self.asymmetry = config.asymmetry

        self.save_epoch = 2
        self.use_wc = config.use_wc
        self.optim = config.optim
        self.clamp_num = config.clamp_num

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

        # pytorch does not have a checkpoint manager
        # have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 25
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name = 'IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

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
        return IM_AE_STATIC.vox_plot(tmpvox, self.local, self.voxel_renderer, self.sampling_threshold, self.real_size)

    def get_voxel_mask_exact(self, vox):
        return IM_AE_STATIC.get_voxel_mask_exact(vox, self.device, self.upsample_rate)

    def recover_voxel(self, vox, xmin, xmax, ymin, ymax, zmin, zmax):
        return recover_voxel(vox, xmin, xmax, ymin, ymax, zmin, zmax,
                             self.real_size, self.upsample_rate, self.mask_margin, self.asymmetry)

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
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            print(" [*] Load SUCCESS")
            self.log.debug(" [*] Load SUCCESS")
            return int(iter_counter)
        else:
            print(f" [!] Load failed... since {checkpoint_txt} doesnt exist")
            self.log.debug(f" [!] Load failed... since {checkpoint_txt} doesnt exist")
            return False

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
                    'discriminator': self.discriminator.state_dict(),
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

    def fine_tune_on_triplet_split(self, config, split):

        if not self.load(config.gpu): exit(-1)

        self.only_discriminator_trainable()

        train_triplet_loader = get_loader(config.data_dir,
                                          os.path.join(config.triplet_splits, f"train_triplets_{split}{config.pct_suffix}.txt"),
                                          batch_size=8)
        test_triplet_loader = get_loader(config.data_dir,
                                         os.path.join(config.triplet_splits, f"test_triplets_{split}.txt"),
                                         batch_size=8)

        criterion = torch.nn.MarginRankingLoss(margin=0.5)
        writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(self.checkpoint_dir), f"finetune_log{config.pct_suffix}"))

        for epoch in range(10):
            train_loss, train_acc = self.fine_tune_epoch(train_triplet_loader, criterion, config.layer)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("accuracy/train", train_acc, epoch)
            test_loss, test_acc = self.test_epoch(test_triplet_loader, criterion, config.layer)
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("accuracy/test", test_acc, epoch)
        print(f"Split {split}, accuracy: {test_acc}")
        self.log.debug(f"Split {split}, accuracy: {test_acc}")
        return test_acc

    def fine_tune_epoch(self, train_triplet_loader, criterion, layer,):
        losses = []
        accuracies = []

        batch_idx = 0
        for batch in iter(train_triplet_loader):
            batch_idx += 1

            batch_loss = 0
            batch_accuracy = 0

            self.discriminator.zero_grad()

            item_idx = 0
            for anchor_path, pos_path, neg_path in zip(*batch):
                item_idx += 1

                anchor_idx = self.dset.files.index(anchor_path)
                pos_idx = self.dset.files.index(pos_path)
                neg_idx = self.dset.files.index(neg_path)

                anchor_dict = self.dset.__getitem__(anchor_idx)
                pos_dict = self.dset.__getitem__(pos_idx)
                neg_dict = self.dset.__getitem__(neg_idx)

                anchor_voxel_style = anchor_dict['voxel_style']
                pos_voxel_style = pos_dict['voxel_style']
                neg_voxel_style = neg_dict['voxel_style']

                anchor_voxel_style = torch.from_numpy(anchor_voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                pos_voxel_style = torch.from_numpy(pos_voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                neg_voxel_style = torch.from_numpy(neg_voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                D_out_anchor = self.discriminator.layer(anchor_voxel_style, layer)
                D_out_pos = self.discriminator.layer(pos_voxel_style, layer)
                D_out_neg = self.discriminator.layer(neg_voxel_style, layer)

                dist_anchor_pos = F.pairwise_distance(D_out_anchor, D_out_pos, 2)
                dist_anchor_neg = F.pairwise_distance(D_out_anchor, D_out_neg, 2)

                target = torch.FloatTensor(dist_anchor_pos.size()).fill_(
                    -1)  # -1 means dist_anchor_neg should be bigger than dist_anchor_pos
                target = Variable(target.to(self.device))
                loss = criterion(dist_anchor_pos, dist_anchor_neg, target)
                batch_loss += loss.item()

                loss.backward(retain_graph=True)

                batch_accuracy += torch.sum(dist_anchor_neg > dist_anchor_pos).item()

            self.optimizer_d.step()
            losses.append(batch_loss)
            batch_accuracy = batch_accuracy / item_idx
            accuracies.append(batch_accuracy)

        return np.mean(losses), np.mean(accuracies)

    def test_epoch(self, test_triplet_loader, criterion, layer):
        accuracies = []
        losses = []
        for batch in iter(test_triplet_loader):
            batch_loss = 0
            batch_accuracy = 0
            for anchor_path, pos_path, neg_path in zip(*batch):
                anchor_idx = self.dset.files.index(anchor_path)
                pos_idx = self.dset.files.index(pos_path)
                neg_idx = self.dset.files.index(neg_path)

                anchor_dict = self.dset.__getitem__(anchor_idx)
                pos_dict = self.dset.__getitem__(pos_idx)
                neg_dict = self.dset.__getitem__(neg_idx)

                anchor_voxel_style = anchor_dict['voxel_style']
                pos_voxel_style = pos_dict['voxel_style']
                neg_voxel_style = neg_dict['voxel_style']

                anchor_voxel_style = torch.from_numpy(anchor_voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                pos_voxel_style = torch.from_numpy(pos_voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                neg_voxel_style = torch.from_numpy(neg_voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                D_out_anchor = self.discriminator.layer(anchor_voxel_style, layer)
                D_out_pos = self.discriminator.layer(pos_voxel_style, layer)
                D_out_neg = self.discriminator.layer(neg_voxel_style, layer)

                dist_anchor_pos = F.pairwise_distance(D_out_anchor, D_out_pos, 2)
                dist_anchor_neg = F.pairwise_distance(D_out_anchor, D_out_neg, 2)

                batch_accuracy += torch.sum(dist_anchor_neg > dist_anchor_pos).item()

                target = torch.FloatTensor(dist_anchor_pos.size()).fill_(
                    -1)  # -1 means dist_anchor_neg should be bigger than dist_anchor_pos
                target = Variable(target.to(self.device))
                loss = criterion(dist_anchor_pos, dist_anchor_neg, target)
                batch_loss += loss.item()
            losses.append(batch_loss)
            batch_accuracy /= len(batch[0])
            accuracies.append(batch_accuracy)

        accuracy = np.mean(accuracies)
        loss = np.mean(losses)
        return loss, accuracy

    def fine_tune_discr_with_rank(self, config):
        splits = len(os.listdir(config.triplet_splits)) // 2
        split_accuracy = {}
        for split in range(splits):
            acc = self.fine_tune_on_triplet_split(config, split)
            split_accuracy[split] = acc

        overall_accuracy = np.sum(list(split_accuracy.values())) / len(split_accuracy)
        overall_accuracy = np.round(overall_accuracy, 3)
        print(f"overall_accuracy: {overall_accuracy}")
        self.log.debug(f"overall_accuracy: {overall_accuracy}")

        checkpoint_dir = os.path.join(config.checkpoint_dir, f"finetune_{config.pct_suffix}", config.layer)
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save({'discriminator': self.discriminator.state_dict(),}, save_path)


class IM_AE_WITH_SE_COMMON(IM_AE_COMMON):

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
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.style_encoder.load_state_dict(checkpoint['style_encoder'])
            print(" [*] Load SUCCESS")
            self.log.debug(" [*] Load SUCCESS")
            return int(iter_counter)
        else:
            print(" [!] Load failed...")
            self.log.debug(" [!] Load failed...")
            return False

    def save(self,iter_counter):
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


def get_building_from_filename(file):
    return os.path.basename(os.path.dirname(os.path.dirname(file)))


def get_element_from_filename(file):
    return os.path.basename(os.path.dirname(file)).split("__")[0].split("_")[-1]

