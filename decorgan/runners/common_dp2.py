import os

from sklearn.manifold import TSNE

from dataset2 import FlexDataset, FlexRndNoiseDataset
from dataset_rot import RotFilesDataset, RotRndNoiseFilesDataset, RotSameFilesDataset
from utils.matplotlib_utils import render_result, render_example, render_views, plot_matrix

from utils import *
from common_utils.utils import normalize_vertices
from modelAE_GD import *
import mcubes
from utils.open3d_utils import TriangleMesh
from utils.open3d_render import render_geometries
from utils import CameraJsonPosition
from PIL import Image
from runners.common import *
from utils.patch_vis import render_generated_with_patches_pytorch, visualize_generated_with_patches_pytorch
import sys

sys.path.extend(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import binvox_rw_faster as binvox_rw


class IM_AE_TWO_DATASETS:

    def _init_data(self, config):

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

    def coarse_detailed_full_plots(self, i, content=True):
        if content:
            data_dict = self.dset.__getitem__(i)
            tmp, _ = self.dset.get_more(i)
        else:
            data_dict = self.style_set.__getitem__(i)
            tmp, _ = self.style_set.get_more(i)

        xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

        # voxel_style = data_dict['voxel_style']
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
            img_coa = render_views([img_coa1, img_coa2], img_size=self.real_size)
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
            img_det_full = render_views([img_det_full1, img_det_full2], img_size=self.real_size)
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

    def visualise_init(self, sample_dir):

        for content_idx in range(5):
            # coarse_img, detailed_smooth_img, detailed_full_img = self.coarse_detailed_full_plots(content_idx)
            coarse_img, detailed_full_img = self.coarse_detailed_full_plots(content_idx)
            title = 'Content'
            render_example([coarse_img,
                           detailed_full_img],
                           save=os.path.join(sample_dir, f"content{content_idx}.png"),
                           title=title,
                           titles=['Coarse',  'Detailed Full'])

        for style_idx in range(0, 5):
            # coarse_img, detailed_smooth_img, detailed_full_img = self.coarse_detailed_full_plots(style_idx)
            coarse_img, detailed_full_img = self.coarse_detailed_full_plots(style_idx)
            title = 'Style'
            render_example([coarse_img,
                           detailed_full_img],
                           save=os.path.join(sample_dir, f"style{style_idx}.png"),
                           title=title,
                           titles=['Coarse', 'Detailed Full'])

    def visualize_styles(self, config):

        iter_size = self.styleset_len
        if config.debug:
            iter_size = 10

        for i in range(iter_size):
            print("preprocessing style - " + str(i + 1) + "/" + str(self.styleset_len))

            data_dict = self.style_set.__getitem__(i)
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
            voxel_style = data_dict['voxel_style']

            img_y = i // 4
            img_x = (i % 4) * 2 + 1
            if img_y < 4:
                tmpvox = self.recover_voxel(voxel_style, xmin, xmax, ymin, ymax, zmin, zmax)
                img = IM_AE_STATIC.vox_plot(tmpvox, self.local, self.voxel_renderer, self.sampling_threshold, self.real_size)
                self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                              img_x * self.real_size:(img_x + 1) * self.real_size] = img
            img_y = i // 4
            img_x = (i % 4) * 2
            if img_y < 4:
                tmp, _ = self.style_set.get_more(i)
                tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                img = IM_AE_STATIC.vox_plot(tmpvox, self.local, self.voxel_renderer, self.sampling_threshold, self.real_size)
                self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                              img_x * self.real_size:(img_x + 1) * self.real_size] = img

    def visualize_contents(self, config):

        iter_size = self.dataset_len
        if config.debug:
            iter_size = 10

        for i in range(iter_size):
            print("preprocessing content - " + str(i + 1) + "/" + str(self.dataset_len))
            self.log.debug("preprocessing content - " + str(i + 1) + "/" + str(self.dataset_len))
            data_dict = self.dset.__getitem__(i)
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

            img_y = i // 4
            img_x = (i % 4) * 2
            if img_y < 4:
                tmp, _ = self.dset.get_more(i)
                tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                img = IM_AE_STATIC.vox_plot(tmpvox, self.local, self.voxel_renderer, self.sampling_threshold, self.real_size)
                self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                              img_x * self.real_size:(img_x + 1) * self.real_size] = img

    def visualize_validation(self, config):

        iter_size = self.valset_len
        if config.debug:
            iter_size = 10

        for i in range(iter_size):
            print("preprocessing validation - " + str(i + 1) + "/" + str(self.valset_len))
            self.log.debug("preprocessing validation - " + str(i + 1) + "/" + str(self.valset_len))
            data_dict = self.vset.__getitem__(i)
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

            img_y = i // 4
            img_x = (i % 4) * 2
            if img_y < 4:
                tmp, _ = self.vset.get_more(i)
                tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                img = IM_AE_STATIC.vox_plot(tmpvox, self.local, self.voxel_renderer, self.sampling_threshold, self.real_size)
                self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                              img_x * self.real_size:(img_x + 1) * self.real_size] = img

    def prepare_voxel_style(self, config):
        # result_dir = "output_for_eval"
        result_dir = config.output_for_eval_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # load style shapes
        self.styleset_names = self.style_set.files

        for style_id in range(self.styleset_len):
            print("preprocessing style - " + str(style_id + 1) + "/" + str(self.styleset_len))
            tmp, _ = self.style_set.get_more(style_id)

            binvox_rw.write_voxel(tmp, result_dir + "/style_" + str(style_id) + ".binvox")

class IM_AE_ONE_DATASET:

    def coarse_detailed_full_plots(self, i):
        data_dict = self.dset.__getitem__(i)
        xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
        tmp, _ = self.dset.get_more(i)

        # voxel_style = data_dict['voxel_style']
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
            img_coa = render_views([img_coa1, img_coa2], img_size=self.real_size)
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
            img_det_full = render_views([img_det_full1, img_det_full2], img_size=self.real_size)
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

    def visualise_init(self, sample_dir):

        for content_idx in range(5):
            content_reference_file = self.dset.get_reference_file(content_idx)
            # coarse_img, detailed_smooth_img, detailed_full_img = self.coarse_detailed_full_plots(content_idx)
            coarse_img, detailed_full_img = self.coarse_detailed_full_plots(content_idx)
            title = 'Content'

            render_example([coarse_img,
                           detailed_full_img],
                           save=os.path.join(sample_dir, f"content{content_idx}.png"),
                           title=title,
                           titles=['Coarse',  'Detailed Full'])


        for style_idx in range(5, 10):
            style_reference_file = self.dset.get_reference_file(style_idx)
            # coarse_img, detailed_smooth_img, detailed_full_img = self.coarse_detailed_full_plots(style_idx)
            coarse_img, detailed_full_img = self.coarse_detailed_full_plots(style_idx)
            title = 'Style'

            render_example([coarse_img,
                           detailed_full_img],
                           save=os.path.join(sample_dir, f"style{style_idx}.png"),
                           title=title,
                           titles=['Coarse', 'Detailed Full'])

    def visualize_content_styles(self, config):

        iter_size = self.dataset_len
        if config.debug:
            iter_size = 10

        for i in range(iter_size):
            print("preprocessing style/content - " + str(i + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(i)
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
            voxel_style = data_dict['voxel_style']

            img_y = i // 4
            img_x = (i % 4) * 2 + 1
            if img_y < 4:
                tmpvox = self.recover_voxel(voxel_style, xmin, xmax, ymin, ymax, zmin, zmax)
                img = IM_AE_STATIC.vox_plot(tmpvox, self.local, self.voxel_renderer, self.sampling_threshold, self.real_size)
                self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                              img_x * self.real_size:(img_x + 1) * self.real_size] = img
            img_y = i // 4
            img_x = (i % 4) * 2
            if img_y < 4:
                tmp, _ = self.dset.get_more(i)
                tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                img = IM_AE_STATIC.vox_plot(tmpvox, self.local, self.voxel_renderer, self.sampling_threshold, self.real_size)
                self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                              img_x * self.real_size:(img_x + 1) * self.real_size] = img


class IM_AE_ORIG_COMMON(IM_AE_COMMON, IM_AE_TWO_DATASETS):

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

                z_vector = np.zeros([self.styleset_len],np.float32)
                z_vector[style_idx] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1,-1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1,-1,1,1,1])
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                # _, detailed_gen_smooth_img, _ = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                (style_idx+1)*self.real_size: (style_idx+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)

    def test_style_codes(self, config):

        self.voxel_renderer.use_gpu(config.gpu)

        if not self.load(config.gpu): exit(-1)
        os.makedirs(config.style_codes_dir, exist_ok=True)

        max_num_of_styles = 64
        max_num_of_styles = min(max_num_of_styles, self.styleset_len)

        style_codes = self.generator.style_codes.detach().cpu().numpy()
        style_codes = (style_codes - np.mean(style_codes, axis=0)) / np.std(style_codes, axis=0)

        style_codes_img = plot_matrix(style_codes, as_img=True)
        cv2.imwrite(config.style_codes_dir + "/" + "style_codes.png", style_codes_img)

        embedded = TSNE(n_components=2, perplexity=16, learning_rate=10.0, n_iter=2000).fit_transform(style_codes)

        print("rendering...")
        img_size = 5000
        if self.styleset_len > 64:
            grid_size = 25
        else:
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
            data_dict = self.style_set.__getitem__(i)
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
            voxel_style = data_dict['voxel_style']
            tmp, _ = self.style_set.get_more(i)
            tmpvox = self.recover_voxel(voxel_style, xmin, xmax, ymin, ymax, zmin, zmax)
            # rendered_view = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvox, self.sampling_threshold)
            vertices, triangles = mcubes.marching_cubes(tmpvox, self.sampling_threshold)
            vertices = normalize_vertices(vertices)
            m_smooth = TriangleMesh(vertices, triangles)
            m_smooth.compute_vertex_normals()
            rendered_view = render_geometries([m_smooth], camera_json=CameraJsonPosition, out_img=True)
            rendered_view = Image.fromarray(np.uint8(np.asarray(rendered_view) * 255))
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

        with torch.no_grad():
            for style_idx in range(self.styleset_len):
                if style_idx not in style_indices:
                    continue

                # _, _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
                _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
                self.imgout_0[0:self.real_size,
                              (style_indices.index(style_idx) + 1)*self.real_size: (style_indices.index(style_idx) + 2)*self.real_size] = detailed_full_style_img

                z_vector = np.zeros([self.styleset_len], np.float32)
                z_vector[style_idx] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])
                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1, -1, 1, 1, 1])

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

    def prepare_voxel_for_eval(self, config):
        # result_dir = "output_for_eval"
        result_dir = config.output_for_eval_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(config.gpu): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 20

        # load style shapes
        self.styleset_names = self.style_set.files

        # load content shapes
        self.dataset_names = self.dset.files
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(content_id)
            tmp_input = data_dict['input']
            tmp_mask = data_dict['mask']

            binvox_rw.write_voxel(tmp_input, result_dir + "/content_" + str(content_id) + "_coarse.binvox")

            mask_fake = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len], np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1, -1, 1, 1, 1])
                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                tmp_voxel_fake = (tmp_voxel_fake > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmp_voxel_fake,
                                      result_dir + "/output_content_" + str(content_id) + "_style_" + str(
                                          style_id) + ".binvox")

    def prepare_voxel_for_FID(self, config):
        # result_dir = "output_for_FID"
        result_dir = config.output_for_FID_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(config.gpu): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 100
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        print(f"Will prepare voxel for fid for {self.dataset_len} contents")

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(content_id)
            tmp_input = data_dict['input']
            tmp_mask = data_dict['mask']
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

            mask_fake = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len], np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])

                z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1, -1, 1, 1, 1])
                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                tmpvox = self.recover_voxel(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                tmpvox = (tmpvox > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmpvox, result_dir + "/output_content_" + str(content_id) + "_style_" + str(
                    style_id) + ".binvox")

    def export(self, config):

        if not self.load(config.gpu): exit(-1)

        with torch.no_grad():

            out_style_dir_all_max = os.path.join(config.export_dir, "discr_all/max")

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
                doutall = self.discriminator.layer(detail_input)

                out_style_file = os.path.join(out_style_dir_all_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, doutall.cpu().numpy().reshape((-1)))

    def freeze_generator(self):
        for name, param in self.generator.named_parameters():
            param.requires_grad = False

    def only_discriminator_trainable(self):
        self.freeze_generator()


class IM_AE_ADJ_COMMON(IM_AE_WITH_SE_COMMON, IM_AE_TWO_DATASETS):

    def visualise(self, sample_dir, epoch):

        self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)

        for style_idx in range(5):
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
            input_content = content_data_dict['input']
            mask_content = content_data_dict['mask']
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
                # _, detailed_gen_smooth_img, _ = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                              (style_idx+1)*self.real_size: (style_idx+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)

    def test_style_codes(self, config):

        self.voxel_renderer.use_gpu(config.gpu)

        if not self.load(config.gpu): exit(-1)
        os.makedirs(config.style_codes_dir, exist_ok=True)

        max_num_of_styles = 64
        max_num_of_styles = min(max_num_of_styles, self.styleset_len)

        style_codes = []
        with torch.no_grad():
            for style_idx in range(0, max_num_of_styles):
                style_data_dict = self.style_set.__getitem__(style_idx)
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
            data_dict = self.style_set.__getitem__(i)
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
            voxel_style = data_dict['voxel_style']

            tmp, _ = self.style_set.get_more(i)
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

        with torch.no_grad():
            for style_idx in range(self.styleset_len):
                if style_idx not in style_indices:
                    continue

                # _, _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
                _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
                self.imgout_0[0:self.real_size,
                              (style_indices.index(style_idx) + 1)*self.real_size: (style_indices.index(style_idx) + 2)*self.real_size] = detailed_full_style_img

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

    def export(self, config):

        if not self.load(config.gpu): exit(-1)

        with torch.no_grad():

            out_style_dir_all_max = os.path.join(config.export_dir, "discr_all/max")
            out_style_dir = os.path.join(config.export_dir, "style_enc_all/max")

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
                z_tensor_g = self.style_encoder.layer(detail_input,  )
                doutall = self.discriminator.layer(detail_input)

                out_style_file = os.path.join(out_style_dir_all_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, doutall.cpu().numpy().reshape((-1)))

                out_style_file = os.path.join(out_style_dir, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, z_tensor_g.cpu().numpy().reshape((-1)))

    def prepare_voxel_for_eval(self, config):
        # result_dir = "output_for_eval"
        result_dir = config.output_for_eval_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(config.gpu): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 20

        # load style shapes
        self.styleset_names = self.style_set.files

        # load content shapes
        self.dataset_names = self.dset.files
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(content_id)
            tmp_input = data_dict['input']
            tmp_mask = data_dict['mask']

            binvox_rw.write_voxel(tmp_input, result_dir + "/content_" + str(content_id) + "_coarse.binvox")

            mask_fake = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):

                voxel_style = self.style_set.__getitem__(style_id)['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1, -1, 1, 1, 1])

                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                tmp_voxel_fake = (tmp_voxel_fake > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmp_voxel_fake,
                                      result_dir + "/output_content_" + str(content_id) + "_style_" + str(
                                          style_id) + ".binvox")

    def prepare_voxel_for_FID(self, config):
        # result_dir = "output_for_FID"
        result_dir = config.output_for_FID_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(config.gpu): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 100
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        print(f"Will prepare voxel for fid for {self.dataset_len} contents")

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(content_id)
            tmp_input = data_dict['input']
            tmp_mask = data_dict['mask']
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

            mask_fake = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):

                voxel_style = self.style_set.__getitem__(style_id)['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1, -1, 1, 1, 1])

                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                tmpvox = self.recover_voxel(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                tmpvox = (tmpvox > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmpvox, result_dir + "/output_content_" + str(content_id) + "_style_" + str(
                    style_id) + ".binvox")

    def freeze_generator(self):
        for name, param in self.generator.named_parameters():
            param.requires_grad = False

    def freeze_style_encoder(self):
        for name, param in self.style_encoder.named_parameters():
            param.requires_grad = False

    def only_discriminator_trainable(self):
        self.freeze_generator()
        self.freeze_style_encoder()


class IM_AE_ADJ_SHARE_COMMON(IM_AE_ADJ_COMMON):

    def export(self, config):

        raise Exception("NOT IMPLEMENTED")


class IM_AE_ADAIN_COMMON(IM_AE_COMMON, IM_AE_TWO_DATASETS):

    def visualise(self, sample_dir, epoch):

        self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)

        for style_idx in range(5):
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
            input_content = content_data_dict['input']
            mask_content = content_data_dict['mask']
            xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_idx in range(5):

                style_data_dict = self.style_set.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']

                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                voxel_fake = self.generator(input_fake,voxel_style,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                # _, detailed_gen_smooth_img, _ = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                              (style_idx+1)*self.real_size: (style_idx+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)

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

        with torch.no_grad():
            for style_idx in range(self.styleset_len):
                if style_idx not in style_indices:
                    continue

                # _, _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
                _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
                self.imgout_0[0:self.real_size,
                              (style_indices.index(style_idx) + 1)*self.real_size: (style_indices.index(style_idx) + 2)*self.real_size] = detailed_full_style_img

                voxel_style = self.style_set.__getitem__(style_idx)['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                for content_idx in range(self.dataset_len):

                    content_data_dict = self.dset.__getitem__(content_idx)
                    mask_content = content_data_dict['mask']
                    input_content = content_data_dict['input']
                    xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

                    mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=False)

                    tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                    # _, detailed_gen_smooth_img, _ = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                    detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)

                    self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size,
                                  (style_indices.index(style_idx)+1)*self.real_size:(style_indices.index(style_idx)+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(config.test_fig_3_dir,f"fig3.png"), self.imgout_0)

    def prepare_voxel_for_eval(self, config):
        # result_dir = "output_for_eval"
        result_dir = config.output_for_eval_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(config.gpu): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 20

        # load style shapes
        self.styleset_names = self.style_set.files

        # load content shapes
        self.dataset_names = self.dset.files
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(content_id)
            tmp_input = data_dict['input']
            tmp_mask = data_dict['mask']

            binvox_rw.write_voxel(tmp_input, result_dir + "/content_" + str(content_id) + "_coarse.binvox")

            mask_fake = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):

                voxel_style = self.style_set.__getitem__(style_id)['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                tmp_voxel_fake = (tmp_voxel_fake > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmp_voxel_fake,
                                      result_dir + "/output_content_" + str(content_id) + "_style_" + str(
                                          style_id) + ".binvox")

    def prepare_voxel_for_FID(self, config):
        # result_dir = "output_for_FID"
        result_dir = config.output_for_FID_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(config.gpu): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 100
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        print(f"Will prepare voxel for fid for {self.dataset_len} contents")

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(content_id)
            tmp_input = data_dict['input']
            tmp_mask = data_dict['mask']
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

            mask_fake = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):

                voxel_style = self.style_set.__getitem__(style_id)['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                tmpvox = self.recover_voxel(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                tmpvox = (tmpvox > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmpvox, result_dir + "/output_content_" + str(content_id) + "_style_" + str(
                    style_id) + ".binvox")

    def export(self, config):
        # if i want multiple layers it's better to iterate over them, otherwise it needs a lot of cuda memory

        if not self.load(config.gpu): exit(-1)

        with torch.no_grad():

            out_style_dir_all_max = os.path.join(config.export_dir, "discr_all/max")
            # out_style_all_dir_max = os.path.join(config.export_dir, "gen_enc_all/max")
            # out_style_last_dir_max = os.path.join(config.export_dir, "gen_enc_last/max")

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
                # z_all = self.generator.layer(detail_input, layer='all')
                # z_tensor_g = self.generator.layer(detail_input, layer='last')
                doutall = self.discriminator.layer(detail_input)

                out_style_file = os.path.join(out_style_dir_all_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, doutall.cpu().numpy().reshape((-1)))

                # out_style_file = os.path.join(out_style_all_dir_max, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, z_all.cpu().numpy().reshape((-1)))

                # out_style_file = os.path.join(out_style_last_dir_max, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, z_tensor_g.cpu().numpy().reshape((-1)))

                # out_style_file = os.path.join(out_style_dir1_max, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, np.max(dout1, (2, 3, 4)))
                # out_style_file = os.path.join(out_style_dir1_avg, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, np.mean(dout1, (2, 3, 4)))

                # out_style_file = os.path.join(out_style_dir2_max, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, np.max(dout2, (2, 3, 4)))
                # out_style_file = os.path.join(out_style_dir2_avg, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, np.mean(dout2, (2, 3, 4)))

                # out_style_file = os.path.join(out_style_dir3_max, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, np.max(dout3, (2, 3, 4)))
                # out_style_file = os.path.join(out_style_dir3_avg, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, np.mean(dout3, (2, 3, 4)))

    def freeze_generator(self):
        for name, param in self.generator.named_parameters():
            param.requires_grad = False

    def only_discriminator_trainable(self):
        self.freeze_generator()


class IM_AE_MYADAIN_COMMON(IM_AE_COMMON, IM_AE_ONE_DATASET):

    def visualise(self, sample_dir, epoch):

        self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)

        for style_idx in range(5, 10):
            _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx)
            self.imgout_0[0:self.real_size,
                          (style_idx-5+1)*self.real_size:(style_idx-5+2)*self.real_size] = detailed_full_style_img

        for content_idx in range(5):

            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx)
            self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                          0:self.real_size] = coarse_content_img

            content_data_dict = self.dset.__getitem__(content_idx)
            input_content = content_data_dict['input']
            mask_content = content_data_dict['mask']
            xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

            mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_idx in range(5, 10):

                style_data_dict = self.dset.__getitem__(style_idx)
                voxel_style = style_data_dict['voxel_style']

                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                voxel_fake = self.generator(input_fake,voxel_style,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                              (style_idx-5+1)*self.real_size: (style_idx-5+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)

    def test_fig_3(self, config):

        if not self.load(config.gpu): exit(-1)
        os.makedirs(config.test_fig_3_dir, exist_ok=True)

        self.imgout_0 = np.full([self.real_size*(3+1),
                                 self.real_size*(11+1)], 255, np.uint8)

        for content_idx in range(3):
            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx)
            self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size,
                          0:self.real_size] = coarse_content_img

        with torch.no_grad():
            for style_idx in range(3, 14):
                _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx)
                self.imgout_0[0:self.real_size,
                              (style_idx-3+1)*self.real_size: (style_idx-3+2)*self.real_size] = detailed_full_style_img

                voxel_style = self.dset.__getitem__(style_idx)['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                for content_idx in range(3):

                    content_data_dict = self.dset.__getitem__(content_idx)
                    input_content = content_data_dict['input']
                    mask_content = content_data_dict['mask']
                    xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

                    mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    voxel_fake = self.generator(input_fake, voxel_style, mask_fake, is_training=False)

                    tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                    detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)

                    self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size,
                                  (style_idx-3+1)*self.real_size:(style_idx-3+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(config.test_fig_3_dir,f"fig3.png"), self.imgout_0)


class IM_AE_ANY_COMMON(IM_AE_ORIG_COMMON, IM_AE_TWO_DATASETS):

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

                z_tensor_g = self.discriminator(voxel_style, is_training=False)
                z_tensor_g = self.discriminator.pool(z_tensor_g)[:, 0:-1, :, :, :]
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                # _, detailed_gen_smooth_img, _ = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
                (style_idx+1)*self.real_size: (style_idx+2)*self.real_size] = detailed_gen_smooth_img

        cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)


class IM_AE_ANY_SHARE_COMMON(IM_AE_COMMON, IM_AE_TWO_DATASETS):

    def visualise(self, sample_dir, epoch):

        self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)

        for style_idx in range(0, 5):
            _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
            self.imgout_0[0:self.real_size,
                          (style_idx+1)*self.real_size:(style_idx+2)*self.real_size] = detailed_full_style_img

        for ci, content_idx in enumerate(range(5, 10)):
            coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx, content=True)
            self.imgout_0[(ci+1)*self.real_size: (ci+2)*self.real_size,
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
                z_tensor_g = self.discriminator_style(d_common_real)
                z_tensor_g = self.discriminator_style.pool(z_tensor_g)
                voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                self.imgout_0[(ci+1)*self.real_size: (ci+2)*self.real_size,
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
            z_tensor_g = self.discriminator_style(d_common_real)
            z_tensor_g = self.discriminator_style.pool(z_tensor_g)

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

        if not self.load(config.gpu): exit(-1)

        with torch.no_grad():

            out_discr_common_global_max = os.path.join(config.export_dir, "discr_common_global/max")
            # out_discr_global_max = os.path.join(config.export_dir, "discr_global/max")
            out_discr_common_style_max = os.path.join(config.export_dir, "discr_common_style/max")
            # out_discr_style_max = os.path.join(config.export_dir, "discr_style/max")
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

                # out_style_file = os.path.join(out_discr_style_max, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, d_style.cpu().numpy().reshape((-1)))

                # out_style_file = os.path.join(out_discr_global_max, building, f"{component}.npy")
                # os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                # np.save(out_style_file, d_global.cpu().numpy().reshape((-1)))

                out = torch.cat([d_common.T, d_global.T, d_style.T])
                out_style_file = os.path.join(out_discr_all_max, building, f"{component}.npy")
                os.makedirs(os.path.dirname(out_style_file), exist_ok=True)
                np.save(out_style_file, out.cpu().numpy().reshape((-1)))

    def prepare_voxel_for_eval(self, config):
        # result_dir = "output_for_eval"
        result_dir = config.output_for_eval_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(config.gpu): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 20

        # load style shapes
        self.styleset_names = self.style_set.files

        # load content shapes
        self.dataset_names = self.dset.files
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(content_id)
            tmp_input = data_dict['input']
            tmp_mask = data_dict['mask']

            binvox_rw.write_voxel(tmp_input, result_dir + "/content_" + str(content_id) + "_coarse.binvox")

            mask_fake = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):

                voxel_style = self.style_set.__getitem__(style_id)['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                d_common = self.common_discriminator(voxel_style, is_training=False)
                z_tensor_g = self.discriminator_style(d_common)

                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                tmp_voxel_fake = (tmp_voxel_fake > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmp_voxel_fake,
                                      result_dir + "/output_content_" + str(content_id) + "_style_" + str(
                                          style_id) + ".binvox")

    def prepare_voxel_for_FID(self, config):
        # result_dir = "output_for_FID"
        result_dir = config.output_for_FID_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load(config.gpu): exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 100
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        print(f"Will prepare voxel for fid for {self.dataset_len} contents")

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))

            data_dict = self.dset.__getitem__(content_id)
            tmp_input = data_dict['input']
            tmp_mask = data_dict['mask']
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']

            mask_fake = torch.from_numpy(tmp_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):

                voxel_style = self.style_set.__getitem__(style_id)['voxel_style']
                voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
                d_common = self.common_discriminator(voxel_style, is_training=False)
                z_tensor_g = self.discriminator_style(d_common)

                voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)

                tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                tmpvox = self.recover_voxel(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                tmpvox = (tmpvox > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmpvox, result_dir + "/output_content_" + str(content_id) + "_style_" + str(
                    style_id) + ".binvox")


class IM_AE_TWO_DATASETS_ROT(IM_AE_TWO_DATASETS):

    def _init_data(self, config):

        print("preprocessing - start")
        self.log.debug("preprocessing - start")

        self.style_set = RotFilesDataset(self.style_dir,
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

        self.dset = RotFilesDataset(self.data_dir,
                                    self.datapath,
                                    dotdict({'cache_dir': config.data_cache_dir,
                                             'input_size': config.input_size,
                                             'output_size': config.output_size,
                                             'asymmetry': config.asymmetry,
                                             'gpu': config.gpu}),
                                    self.log,
                                    filename=config.data_filename)
        self.dataset_len = len(self.dset)  # this is ok not to be constant

        if (config.train and not config.debug) or config.visualize_contents:
            self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
            self.visualize_contents(config)
            cv2.imwrite(config.sample_dir + "/a_content_0.png", self.imgout_0)

        self.valpath = config.valpath
        self.val_dir = config.val_dir
        self.vset = RotFilesDataset(self.val_dir,
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


class IM_AE_ANY_SHARE_COMMON_ROT(IM_AE_ANY_SHARE_COMMON, IM_AE_TWO_DATASETS_ROT):
    pass


class IM_AE_TWO_DATASETS_ROT_SAME(IM_AE_TWO_DATASETS):

    def _init_data(self, config):

        print("preprocessing - start")
        self.log.debug("preprocessing - start")

        self.style_set = RotSameFilesDataset(self.style_dir,
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

        self.dset = RotSameFilesDataset(self.data_dir,
                                        self.datapath,
                                        dotdict({'cache_dir': config.data_cache_dir,
                                                 'input_size': config.input_size,
                                                 'output_size': config.output_size,
                                                 'asymmetry': config.asymmetry,
                                                 'gpu': config.gpu}),
                                        self.log,
                                        filename=config.data_filename)
        self.dataset_len = len(self.dset)  # this is ok not to be constant

        if (config.train and not config.debug) or config.visualize_contents:
            self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
            self.visualize_contents(config)
            cv2.imwrite(config.sample_dir + "/a_content_0.png", self.imgout_0)

        self.valpath = config.valpath
        self.val_dir = config.val_dir
        self.vset = RotSameFilesDataset(self.val_dir,
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


class IM_AE_ANY_SHARE_COMMON_ROT_SAME(IM_AE_ANY_SHARE_COMMON, IM_AE_TWO_DATASETS_ROT_SAME):

    def visualise(self, sample_dir, epoch):

        self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)

        for rot in [0, 108, 216]:
            possible_indices = self.style_set.get_file_indices_with_rot(rot)
            random.shuffle(possible_indices)

            for si, style_idx in enumerate(possible_indices[0:5]):
                _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
                self.imgout_0[0:self.real_size,
                              (si+1)*self.real_size:(si+2)*self.real_size] = detailed_full_style_img

            for ci, content_idx in enumerate(possible_indices[5:10]):
                coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx, content=True)
                self.imgout_0[(ci+1)*self.real_size: (ci+2)*self.real_size,
                              0:self.real_size] = coarse_content_img

                content_data_dict = self.dset.__getitem__(content_idx)
                mask_content = content_data_dict['mask']
                input_content = content_data_dict['input']
                xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']

                mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()

                for si, style_idx in enumerate(possible_indices[0:5]):
                        style_data_dict = self.style_set.__getitem__(style_idx)
                        voxel_style = style_data_dict['voxel_style']

                        voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)

                        d_common_real = self.common_discriminator(voxel_style, is_training=False)
                        z_tensor_g = self.discriminator_style(d_common_real)
                        z_tensor_g = self.discriminator_style.pool(z_tensor_g)
                        voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)

                        tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
                        detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                        self.imgout_0[(ci+1)*self.real_size: (ci+2)*self.real_size,
                                      (si+1)*self.real_size: (si+2)*self.real_size] = detailed_gen_smooth_img

            cv2.imwrite(os.path.join(sample_dir, f"{epoch}_{rot}.png"), self.imgout_0)


class IM_AE_TWO_DATASETS_ROT_RNDNOISE(IM_AE_TWO_DATASETS):

    def _init_data(self, config):

        print("preprocessing - start")
        self.log.debug("preprocessing - start")

        self.style_set = RotRndNoiseFilesDataset(self.style_dir,
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

        self.dset = RotRndNoiseFilesDataset(self.data_dir,
                                            self.datapath,
                                            dotdict({'cache_dir': config.data_cache_dir,
                                                     'input_size': config.input_size,
                                                     'output_size': config.output_size,
                                                     'asymmetry': config.asymmetry,
                                                     'gpu': config.gpu}),
                                            self.log,
                                            filename=config.data_filename)
        self.dataset_len = len(self.dset)  # this is ok not to be constant

        if (config.train and not config.debug) or config.visualize_contents:
            self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
            self.visualize_contents(config)
            cv2.imwrite(config.sample_dir + "/a_content_0.png", self.imgout_0)

        self.valpath = config.valpath
        self.val_dir = config.val_dir
        self.vset = RotRndNoiseFilesDataset(self.val_dir,
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


class IM_AE_TWO_DATASETS_RNDNOISE(IM_AE_TWO_DATASETS):

    def _init_data(self, config):

        print("preprocessing - start")
        self.log.debug("preprocessing - start")

        self.style_set = FlexRndNoiseDataset(self.style_dir,
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

        self.dset = FlexRndNoiseDataset(self.data_dir,
                                        self.datapath,
                                        dotdict({'cache_dir': config.data_cache_dir,
                                                 'input_size': config.input_size,
                                                 'output_size': config.output_size,
                                                 'asymmetry': config.asymmetry,
                                                 'gpu': config.gpu}),
                                        self.log,
                                        filename=config.data_filename)
        self.dataset_len = len(self.dset)  # this is ok not to be constant

        if (config.train and not config.debug) or config.visualize_contents:
            self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
            self.visualize_contents(config)
            cv2.imwrite(config.sample_dir + "/a_content_0.png", self.imgout_0)

        self.valpath = config.valpath
        self.val_dir = config.val_dir
        self.vset = FlexRndNoiseDataset(self.val_dir,
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


# class IM_AE_MYMODEL_COMMON(IM_AE_WITH_SE_COMMON, IM_AE_ONE_DATASET):
#
#     def visualise(self, sample_dir, epoch):
#
#         self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)], 255, np.uint8)
#
#         for style_idx in range(5, 10):
#             _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx)
#             self.imgout_0[0:self.real_size,
#                           (style_idx-5+1)*self.real_size:(style_idx-5+2)*self.real_size] = detailed_full_style_img
#
#         for content_idx in range(5):
#
#             coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx)
#             self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
#                           0:self.real_size] = coarse_content_img
#
#             content_data_dict = self.dset.__getitem__(content_idx)
#             input_content = content_data_dict['input']
#             mask_content = content_data_dict['mask']
#             xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']
#
#             mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#             input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#
#         for style_idx in range(5, 10):
#
#                 style_data_dict = self.dset.__getitem__(style_idx)
#                 voxel_style = style_data_dict['voxel_style']
#
#                 voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
#
#                 z_tensor_g = self.style_encoder(voxel_style, is_training=False)
#                 voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)
#
#                 tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
#                 detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
#                 self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
#                               (style_idx-5+1)*self.real_size: (style_idx-5+2)*self.real_size] = detailed_gen_smooth_img
#
#         cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)
#
#     def test_style_codes(self, config):
#
#         # self.voxel_renderer.use_gpu()
#
#         if not self.load(config.gpu): exit(-1)
#         os.makedirs(config.style_codes_dir, exist_ok=True)
#
#         max_num_of_styles = 64
#         max_num_of_styles = min(max_num_of_styles, len(self.dset))
#
#         style_codes = []
#         with torch.no_grad():
#             for style_idx in range(0, max_num_of_styles):
#                 style_data_dict = self.dset.__getitem__(style_idx)
#                 voxel_style = style_data_dict['voxel_style']
#                 voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
#                 z_tensor_g = self.style_encoder(voxel_style, is_training=False)
#                 style_codes.append(z_tensor_g.detach().cpu().numpy()[0, :, 0, 0, 0])
#         style_codes = np.vstack(style_codes)
#         # style_codes = (style_codes - np.mean(style_codes, axis=0)) / np.std(style_codes, axis=0)
#
#         style_codes_img = plot_matrix(style_codes, as_img=True)
#         cv2.imwrite(config.style_codes_dir + "/" + "style_codes.png", style_codes_img)
#
#         embedded = TSNE(n_components=2,perplexity=16,learning_rate=10.0,n_iter=2000).fit_transform(style_codes)
#
#         print("rendering...")
#         img_size = 5000
#         grid_size = 20
#         if self.output_size == 128:
#             cell_size = 140
#         elif self.output_size == 256:
#             cell_size = 180
#         plt = np.full([img_size + self.real_size, img_size + self.real_size], 255, np.uint8)
#         plt_grid = np.full([grid_size * cell_size + (self.real_size - cell_size),
#                             grid_size * cell_size + (self.real_size - cell_size)], 255, np.uint8)
#         occ_grid = np.zeros([grid_size, grid_size], np.uint8)
#
#         x_max = np.max(embedded[:, 0])
#         x_min = np.min(embedded[:, 0])
#         y_max = np.max(embedded[:, 1])
#         y_min = np.min(embedded[:, 1])
#         x_mid = (x_max + x_min) / 2
#         y_mid = (y_max + y_min) / 2
#         scalex = (x_max - x_min) * 1.05
#         scaley = (y_max - y_min) * 1.05
#         embedded[:, 0] = ((embedded[:, 0] - x_mid) / scalex + 0.5) * img_size
#         embedded[:, 1] = ((embedded[:, 1] - y_mid) / scaley + 0.5) * img_size
#
#         for i in range(max_num_of_styles):
#             data_dict = self.dset.__getitem__(i)
#             xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
#             voxel_style = data_dict['voxel_style']
#             tmpvox = self.recover_voxel(voxel_style, xmin, xmax, ymin, ymax, zmin, zmax)
#             # rendered_view = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvox, self.sampling_threshold)
#             vertices, triangles = mcubes.marching_cubes(tmpvox, self.sampling_threshold)
#             vertices = normalize_vertices(vertices)
#             m_smooth = TriangleMesh(vertices, triangles)
#             m_smooth.compute_vertex_normals()
#             rendered_view = render_geometries([m_smooth], camera_json=CameraJsonPosition, out_img=True)
#             rendered_view = Image.fromarray(np.uint8(np.asarray(rendered_view)*255))
#             rendered_view = np.asarray(rendered_view.resize((self.real_size, self.real_size)).convert('L'))
#
#             img_x = int(embedded[i, 0])
#             img_y = int(embedded[i, 1])
#             plt[img_y:img_y + self.real_size, img_x:img_x + self.real_size] = np.minimum(
#                 plt[img_y:img_y + self.real_size, img_x:img_x + self.real_size], rendered_view)
#
#             img_x = int(embedded[i, 0] / img_size * grid_size)
#             img_y = int(embedded[i, 1] / img_size * grid_size)
#             if occ_grid[img_y, img_x] == 0:
#                 img_y = img_y
#                 img_x = img_x
#             elif img_y - 1 >= 0 and occ_grid[img_y - 1, img_x] == 0:
#                 img_y = img_y - 1
#                 img_x = img_x
#             elif img_y + 1 < grid_size and occ_grid[img_y + 1, img_x] == 0:
#                 img_y = img_y + 1
#                 img_x = img_x
#             elif img_x - 1 >= 0 and occ_grid[img_y, img_x - 1] == 0:
#                 img_y = img_y
#                 img_x = img_x - 1
#             elif img_x + 1 < grid_size and occ_grid[img_y, img_x + 1] == 0:
#                 img_y = img_y
#                 img_x = img_x + 1
#             elif img_y - 1 >= 0 and img_x - 1 >= 0 and occ_grid[img_y - 1, img_x - 1] == 0:
#                 img_y = img_y - 1
#                 img_x = img_x - 1
#             elif img_y + 1 < grid_size and img_x - 1 >= 0 and occ_grid[img_y + 1, img_x - 1] == 0:
#                 img_y = img_y + 1
#                 img_x = img_x - 1
#             elif img_y - 1 >= 0 and img_x + 1 < grid_size and occ_grid[img_y - 1, img_x + 1] == 0:
#                 img_y = img_y - 1
#                 img_x = img_x + 1
#             elif img_y + 1 < grid_size and img_x + 1 < grid_size and occ_grid[img_y + 1, img_x + 1] == 0:
#                 img_y = img_y + 1
#                 img_x = img_x + 1
#             else:
#                 print("warning: cannot find spot")
#             occ_grid[img_y, img_x] = 1
#             img_x *= cell_size
#             img_y *= cell_size
#             plt_grid[img_y:img_y + self.real_size, img_x:img_x + self.real_size] = np.minimum(
#                 plt_grid[img_y:img_y + self.real_size, img_x:img_x + self.real_size], rendered_view)
#
#         cv2.imwrite(config.style_codes_dir + "/" + "latent_gz.png", plt)
#         cv2.imwrite(config.style_codes_dir + "/" + "latent_gz_grid.png", plt_grid)
#         print("rendering...complete")
#
#     def test_fig_3(self, config):
#
#         if not self.load(config.gpu): exit(-1)
#         os.makedirs(config.test_fig_3_dir, exist_ok=True)
#
#         self.imgout_0 = np.full([self.real_size*(3+1),
#                                  self.real_size*(11+1)], 255, np.uint8)
#
#         for content_idx in range(3):
#             coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx)
#             self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size,
#                           0:self.real_size] = coarse_content_img
#
#         for style_idx in range(3, 14):
#             _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx)
#             self.imgout_0[0:self.real_size,
#                           (style_idx-3+1)*self.real_size: (style_idx-3+2)*self.real_size] = detailed_full_style_img
#
#             voxel_style = self.dset.__getitem__(style_idx)['voxel_style']
#             voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
#             z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1, -1, 1, 1, 1])
#
#             for content_idx in range(3):
#
#                 content_data_dict = self.dset.__getitem__(content_idx)
#                 input_content = content_data_dict['input']
#                 mask_content = content_data_dict['mask']
#                 xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']
#
#                 mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#                 input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#
#                 voxel_fake = self.generator(input_fake, z_tensor_g, mask_fake, is_training=False)
#
#                 tmp_voxel_fake = voxel_fake.detach().cpu().numpy()[0, 0]
#                 # _, detailed_gen_smooth_img, _ = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
#                 detailed_gen_smooth_img = self.generation_plot(tmp_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
#
#                 self.imgout_0[(content_idx+1)*self.real_size:(content_idx+2)*self.real_size,
#                               (style_idx-3+1)*self.real_size:(style_idx-3+2)*self.real_size] = detailed_gen_smooth_img
#
#         cv2.imwrite(os.path.join(config.test_fig_3_dir,f"fig3.png"), self.imgout_0)
#
#
# class IM_AE_ADJ_PATCHD_COMMON(IM_AE_ADJ_COMMON):
#
#     def visualise(self, sample_dir, epoch, get_random_paired_patches_f):
#
#         self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(3+1)*2], 255, np.uint8)
#
#         for style_idx in range(3):
#             _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
#             self.imgout_0[0:self.real_size,
#                           (style_idx*2+1)*self.real_size:(style_idx*2+2)*self.real_size] = detailed_full_style_img
#
#         for content_idx in range(5):
#
#             coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx, content=True)
#             self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
#                           0:self.real_size] = coarse_content_img
#
#             content_data_dict = self.dset.__getitem__(content_idx)
#             input_content = content_data_dict['input']
#             mask_content = content_data_dict['mask']
#             xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']
#             voxel_content, _ = self.dset.get_more(content_idx)
#
#             mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#             input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#             voxel_content = torch.from_numpy(voxel_content).to(self.device)
#
#             for style_idx in range(3):
#
#                 style_data_dict = self.style_set.__getitem__(style_idx)
#                 voxel_style = style_data_dict['voxel_style']
#
#                 voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
#
#                 z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1,-1,1,1,1])
#                 voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)
#
#                 tmp_voxel_fake = voxel_fake[0, 0]
#                 voxel_content_patches, voxel_fake_patches, xyz_patches = get_random_paired_patches_f(
#                     voxel_content, tmp_voxel_fake,
#                     patch_size=self.patch_size, patch_num=self.patch_num, stride=self.patch_stride)
#                 if voxel_content_patches.shape[0] == 0:
#                     continue
#
#                 if self.local:
#                     self.imgout_0[(content_idx + 1) * self.real_size: (content_idx + 2) * self.real_size,
#                                   (style_idx * 2 + 1) * self.real_size: (style_idx * 2 + 3) * self.real_size] = \
#                         visualize_generated_with_patches_pytorch(tmp_voxel_fake, voxel_fake_patches, xyz_patches,
#                                                                  thr=self.sampling_threshold)
#                 else:
#                     self.imgout_0[(content_idx + 1) * self.real_size: (content_idx + 2) * self.real_size,
#                                   (style_idx*2 + 1) * self.real_size: (style_idx*2 + 3) * self.real_size] = \
#                         render_generated_with_patches_pytorch(tmp_voxel_fake, voxel_fake_patches, xyz_patches,
#                                                               (xmin, xmax, ymin, ymax, zmin, zmax),
#                                                               thr=self.sampling_threshold,
#                                                               vert=False)
#
#         cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)
#
#
# class IM_AE_ADJ_NONCUBE_PATCHD_COMMON(IM_AE_ADJ_COMMON):
#
#     def visualise(self, sample_dir, epoch, get_random_paired_patches_f):
#
#         self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(3+1)*2], 255, np.uint8)
#
#         for style_idx in range(3):
#             _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx, content=False)
#             self.imgout_0[0:self.real_size,
#                           (style_idx*2+1)*self.real_size:(style_idx*2+2)*self.real_size] = detailed_full_style_img
#
#         for content_idx in range(5):
#
#             coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx, content=True)
#             self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
#                           0:self.real_size] = coarse_content_img
#
#             content_data_dict = self.dset.__getitem__(content_idx)
#             input_content = content_data_dict['input']
#             mask_content = content_data_dict['mask']
#             xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']
#             voxel_content, _ = self.dset.get_more(content_idx)
#
#             mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#             input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#             voxel_content = torch.from_numpy(voxel_content).to(self.device)
#
#             for style_idx in range(3):
#
#                 style_data_dict = self.style_set.__getitem__(style_idx)
#                 voxel_style = style_data_dict['voxel_style']
#
#                 voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
#
#                 z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1,-1,1,1,1])
#                 voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)
#
#                 tmp_voxel_fake = voxel_fake[0, 0]
#                 voxel_content_patches, voxel_fake_patches, xyz_patches = get_random_paired_patches_f(
#                     voxel_content, tmp_voxel_fake,
#                     patch_factor=self.patch_factor, patch_num=self.patch_num, stride_factor=self.stride_factor)
#                 if voxel_content_patches.shape[0] == 0:
#                     continue
#
#                 if self.local:
#                     self.imgout_0[(content_idx + 1) * self.real_size: (content_idx + 2) * self.real_size,
#                                   (style_idx * 2 + 1) * self.real_size: (style_idx * 2 + 3) * self.real_size] = \
#                         visualize_generated_with_patches_pytorch(tmp_voxel_fake, voxel_fake_patches, xyz_patches,
#                                                                  thr=self.sampling_threshold)
#                 else:
#                     self.imgout_0[(content_idx + 1) * self.real_size: (content_idx + 2) * self.real_size,
#                                   (style_idx*2 + 1) * self.real_size: (style_idx*2 + 3) * self.real_size] = \
#                         render_generated_with_patches_pytorch(tmp_voxel_fake, voxel_fake_patches, xyz_patches,
#                                                               (xmin, xmax, ymin, ymax, zmin, zmax),
#                                                               thr=self.sampling_threshold,
#                                                               vert=False)
#
#         cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)
#
#
# class IM_AE_MYMODEL_NONCUBE_PATCHD_COMMON(IM_AE_MYMODEL_COMMON):
#
#     def visualise(self, sample_dir, epoch, get_random_paired_patches_f):
#
#         self.imgout_0 = np.full([self.real_size*(5+1), self.real_size*(5+1)*2], 255, np.uint8)
#
#         for style_idx in range(5,10):
#             _, detailed_full_style_img = self.coarse_detailed_full_plots(style_idx)
#             self.imgout_0[0:self.real_size,
#                           ((style_idx-5)*2+1)*self.real_size:((style_idx-5)*2+2)*self.real_size] = detailed_full_style_img
#
#         for content_idx in range(5):
#
#             coarse_content_img, _ = self.coarse_detailed_full_plots(content_idx)
#             self.imgout_0[(content_idx+1)*self.real_size: (content_idx+2)*self.real_size,
#                           0:self.real_size] = coarse_content_img
#
#             content_data_dict = self.dset.__getitem__(content_idx)
#             input_content = content_data_dict['input']
#             mask_content = content_data_dict['mask']
#             xmin, xmax, ymin, ymax, zmin, zmax = content_data_dict['pos']
#             voxel_content, _ = self.dset.get_more(content_idx)
#
#             mask_fake = torch.from_numpy(mask_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#             input_fake = torch.from_numpy(input_content).to(self.device).unsqueeze(0).unsqueeze(0).float()
#             voxel_content = torch.from_numpy(voxel_content).to(self.device)
#
#             for style_idx in range(5,10):
#
#                 style_data_dict = self.dset.__getitem__(style_idx)
#                 voxel_style = style_data_dict['voxel_style']
#
#                 voxel_style = torch.from_numpy(voxel_style).to(self.device).unsqueeze(0).unsqueeze(0)
#
#                 z_tensor_g = self.style_encoder(voxel_style, is_training=False).view([1,-1,1,1,1])
#                 voxel_fake = self.generator(input_fake,z_tensor_g,mask_fake,is_training=False)
#
#                 tmp_voxel_fake = voxel_fake[0, 0]
#                 voxel_content_patches, voxel_fake_patches, xyz_patches = get_random_paired_patches_f(
#                     voxel_content, tmp_voxel_fake,
#                     patch_factor=self.patch_factor, patch_num=self.patch_num, stride_factor=self.stride_factor)
#                 if voxel_content_patches.shape[0] == 0:
#                     continue
#
#                 if self.local:
#                     img = visualize_generated_with_patches_pytorch(tmp_voxel_fake, voxel_fake_patches, xyz_patches,
#                                                                    thr=self.sampling_threshold)
#                     img = Image.fromarray(np.uint8(np.asarray(img) * 255))
#                     img = np.asarray(img.resize((self.real_size*2, self.real_size)).convert('L'))
#                     self.imgout_0[(content_idx + 1) * self.real_size: (content_idx + 2) * self.real_size,
#                                   ((style_idx-5) * 2 + 1) * self.real_size: ((style_idx-5) * 2 +3) * self.real_size] = img
#
#                 else:
#                     self.imgout_0[(content_idx + 1) * self.real_size: (content_idx + 2) * self.real_size,
#                                   ((style_idx-5)*2 + 1) * self.real_size: ((style_idx-5)*2 + 3) * self.real_size] = \
#                         render_generated_with_patches_pytorch(tmp_voxel_fake, voxel_fake_patches, xyz_patches,
#                                                               (xmin, xmax, ymin, ymax, zmin, zmax),
#                                                               thr=self.sampling_threshold,
#                                                               vert=False)
#
#         cv2.imwrite(os.path.join(sample_dir,f"{epoch}.png"), self.imgout_0)
