import os
import unittest
from unittest import TestCase

from torch.utils.data import DataLoader

from dataset2 import BasicDataset, FlexDataset
import logging
import numpy as np
import mcubes

from runners.common import IM_AE_STATIC
from common_utils.utils import normalize_vertices, get_points_from_voxel
from utils import get_bound_points_from_voxel, dotdict, get_vox_from_binvox_1over2, recover_voxel
from utils.matplotlib_utils import fig_to_img
from utils.pytorch3d_vis import CustomMultiViewMeshRenderer, join_voxel_and_bbox
from utils.open3d_vis import interactive_plot
from utils.open3d_utils import TriangleMesh, PointCloud, get_unit_bbox
import torch.nn.functional as F
import torch
from utils.open3d_render import render_geometries
from utils import CameraJsonPosition
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch


logger = logging.getLogger(__name__)


cmvr = CustomMultiViewMeshRenderer(torch.device('cuda:0'))


class TestData(TestCase):

    real_size = 256
    mask_margin = 8
    upsample_rate = None
    asymmetry = True
    device = torch.device('cuda')
    sampling_threshold = 0.4

    def get_voxel_bbox(self, vox, debug=False):
        # minimap
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate,
                                         padding=0)
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0, 0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        # x
        ray = np.max(smallmaskx, (1, 2))
        indices = np.where(ray == 1)
        xmin = indices[0][0]
        xmax = indices[0][-1]
        # y
        ray = np.max(smallmaskx, (0, 2))
        indices = np.where(ray == 1)
        ymin = indices[0][0]
        ymax = indices[0][-1]
        # z
        ray = np.max(smallmaskx, (0, 1))
        if self.asymmetry:
            indices = np.where(ray == 1)
            zmin = indices[0][0]
            zmax = indices[0][-1]
        else:
            raise Exception("cant")
        if debug:
            return xmin, xmax + 1, ymin, ymax + 1, zmin, zmax + 1, smallmaskx
        return xmin, xmax + 1, ymin, ymax + 1, zmin, zmax + 1

    def test_shapes(self):

        element = "door"
        styles = 16
        data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components"
        fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/groups_june17_uni_nor_components_{element}_styles{styles}.txt"
        config = dotdict({"asymmetry": True, 'gpu': 0})
        dset = BasicDataset(data_dir, fpath, config, logger, filename='model_filled.binvox')
        print(f"dset length: {len(dset)}")

        for i in range(len(dset)):
            data_dict = dset.__getitem__(i)

            input = data_dict['input']
            mask = data_dict['mask']
            Dmask = data_dict['Dmask']

            voxel_style = data_dict['voxel_style']
            Dmask_style = data_dict['Dmask_style']

            binvox_file = dset.files[i]
            name = os.path.dirname(binvox_file)

            voxel_centers_input = get_points_from_voxel(input)
            voxel_centers_input += np.array([-2, 0, 0])
            voxel_centers_voxel_style = get_points_from_voxel(voxel_style)
            voxel_centers_voxel_style += np.array([-1, 0, 0])

            vertices, triangles = mcubes.marching_cubes(input, 0)
            vertices = normalize_vertices(vertices)
            vertices += np.array([1, 0, 0])
            m_input = TriangleMesh(vertices, triangles)
            m_input.compute_vertex_normals()

            vertices, triangles = mcubes.marching_cubes(voxel_style, 0)
            vertices = normalize_vertices(vertices)
            vertices += np.array([2., 0, 0])
            m_voxel_style = TriangleMesh(vertices, triangles)
            m_voxel_style.compute_vertex_normals()
            render_geometries([PointCloud(voxel_centers_input), PointCloud(voxel_centers_voxel_style),
                               m_input, m_voxel_style],
                              window_name=name,
                              camera_json=CameraJsonPosition)

            # tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
            # print(f"tmp_raw shape {tmp_raw.shape}")
            # print(f"mask shape {mask.shape}")
            # print(f"Dmask shape {Dmask.shape}")
            # print(f"Dmask_style shape {Dmask_style.shape}")

            # xmin, xmax, ymin, ymax, zmin, zmax, pooled_tmp_raw = dset.get_voxel_bbox(tmp_raw, debug=True)
            # cropped = dset.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            # all_ones_tmp_raw = np.ones_like(tmp_raw)
            # all_ones_cropped = np.ones_like(cropped)

            # voxel_centers_pooled_tmp_raw = get_points_from_voxel(pooled_tmp_raw)
            # bound_points_pooled_tmp_raw = get_bound_points_from_voxel(pooled_tmp_raw)
            # reds = np.array([[255,0,0]]*len(bound_points_pooled_tmp_raw))
            # blues = np.array([[0,0,255]]*len(voxel_centers_pooled_tmp_raw))
            # interactive_plot([PointCloud(bound_points_pooled_tmp_raw, reds),
            #                   PointCloud(voxel_centers_pooled_tmp_raw, blues)])

            # genika to Dmask_style xrisimopoieite sto loss_d_real (me to discriminator output given voxel_style)
            # kai to Dmask (content) xrisimopoieite sto loss_d_fake i loss_g (me to discriminator output given fake)
            # kai to mask (content) xrisimopoieite sto generation given content kai style
            # opotan den me noiazei kanonika to height kai width tou kathe shape !!!

    def coarse_detailed(self, data_dir, fpath, config, out_dir, max_iter=200):
        dset = FlexDataset(data_dir, fpath, config, logger, filename=config['filename'])
        print(f"dset length: {len(dset)}")

        self.upsample_rate = dset.upsample_rate

        imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)

        i = -1
        for j in range(min(len(dset), max_iter)):
            i += 1
            data_dict = dset.__getitem__(j)
            xmin, xmax, ymin, ymax, zmin, zmax = data_dict['pos']
            voxel_style = data_dict['voxel_style']

            img_y = i // 4
            img_x = (i % 4) * 2 + 1

            tmpvox = recover_voxel(voxel_style, xmin, xmax, ymin, ymax, zmin, zmax,
                                   self.real_size, self.upsample_rate, self.mask_margin, self.asymmetry)
            img = IM_AE_STATIC.vox_plot(tmpvox, True, None, self.sampling_threshold, self.real_size)
            imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                     img_x * self.real_size:(img_x + 1) * self.real_size] = img

            img_y = i // 4
            img_x = (i % 4) * 2

            tmp, _ = dset.get_more(j)
            tmp_mask_exact = IM_AE_STATIC.get_voxel_mask_exact(tmp, self.device, self.upsample_rate)
            tmpvox = recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax,
                                   self.real_size, self.upsample_rate, self.mask_margin, self.asymmetry)
            img = IM_AE_STATIC.vox_plot(tmpvox, True, None, self.sampling_threshold, self.real_size)
            imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                     img_x * self.real_size:(img_x + 1) * self.real_size] = img

            if (i % 15 == 0 and i > 0) or j == (len(dset) - 1):
                os.makedirs(out_dir, exist_ok=True)
                if "style" not in fpath:
                    cv2.imwrite(f"{out_dir}/content{j}.png", imgout_0)
                else:
                    cv2.imwrite(f"{out_dir}/style{j}.png", imgout_0)
                imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)
                i = -1

    def test_coarse_detailed_component(self):
        element = "tower_steepledomedoorwindowcolumn"
        split = "train"
        out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local/check_{element}"
        data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components"
        fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/groups_june17_uni_nor_components_{element}_{split}.txt"
        config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 16, 'output_size': 128, "filename": 'model_filled.binvox'})
        self.coarse_detailed(data_dir, fpath, config, out_dir)

    def test_coarse_detailed_building_i16o128(self):
        out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local/check_building/setA/i16o128"
        data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_buildings/normalizedObj"
        # fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/buildnet_buildings/buildnet_annfass_buildings_setA.txt"
        fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/buildnet_buildings/buildnet_buildings_setAstyles32.txt"
        config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 16, 'output_size': 128, "filename":'model_filled.binvox'})
        self.coarse_detailed(data_dir, fpath, config, out_dir)

    def test_coarse_detailed_yudata_i16o128(self):

        out_dir = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/HU_YU_LUN_BUILDNET/yu_data/chair_yu_norm_and_align"
        data_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/preprocessed_data/chair_yu_norm_and_align"
        fpath = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/splits/chair_yu_norm_and_align/export.txt"

        config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 16, 'output_size': 128, "filename": "model_filled.binvox"})
        self.coarse_detailed(data_dir, fpath, config, out_dir, max_iter=33)

    def test_coarse_detailed_hudata_i16o128(self):

        out_dir = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/visualize_data/hu_data/building_hu_tri_norm_and_align"
        data_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/preprocessed_data/building_hu_tri_norm_and_align"
        fpath = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/splits/building_hu_tri_norm_and_align/export.txt"

        # config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 16, 'output_size': 128, "filename": "model_filled.binvox"})
        config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 32, 'output_size': 256, "filename": "model_filled.binvox"})
        self.coarse_detailed(data_dir, fpath, config, out_dir, max_iter=33)

    def test_coarse_detailed_building_i32o256(self):
        out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local/check_building/i32o256"
        data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_buildings/normalizedObj"
        fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/buildnet_buildings_styles_16.txt"
        config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 32, 'output_size': 256, "filename": "model_filled.binvox"})
        self.coarse_detailed(data_dir, fpath, config, out_dir)

    def test_coarse_detailed_car_i32o256(self):
        out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local/check_car/i32o256"
        data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/02958343"
        # fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/car/style_car_64.txt"
        fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/car/content_car_train.txt"
        config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 32, 'output_size': 256, "filename":'model_depth_fusion.binvox'})
        self.coarse_detailed(data_dir, fpath, config, out_dir)

    def test_coarse_detailed_chair_selection(self):
        # out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local/check_chair/random_references_styles"
        # out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local/check_chair/styles_64"
        out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local/check_chair/random_rot"
        # data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/03001627"
        data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/preprocessed_data/03001627_train_objs_norm_and_random_rot"
        # fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/random_style_references_64.txt"
        # fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/style_chair_64.txt"
        fpath = f"/home/graphicslab/Desktop/todel.txt"
        # config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 32, 'output_size': 256, "filename":'model_depth_fusion.binvox'})
        config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 32, 'output_size': 256, "filename":'model_filled.binvox'})
        self.coarse_detailed(data_dir, fpath, config, out_dir,)

    def test_shapes_in_boxes(self):

        element = "door"
        styles = 16
        data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components"
        fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/groups_june17_uni_nor_components_{element}_styles{styles}.txt"
        config = dotdict({"asymmetry": True, 'gpu': 0})
        dset = BasicDataset(data_dir, fpath, config, logger, filename='model_filled.binvox')
        print(f"dset length: {len(dset)}")

        for i in range(len(dset)):

            geometries = []

            data_dict = dset.__getitem__(i)

            input = data_dict['input']  # coarse shape (pooling with stride same as kernel i.e. upsample rate)
            mask = data_dict['mask']  # mask on coarse shape (this is to be used in the generator) thus it must
                                      # be the same as detailed shape but its not? :/
            Dmask = data_dict['Dmask']  # dilated mask on coarse shape (this is to be used with discriminator's output,
                                        # therefore it's size is half the detailed shape. we basically just upsample
                                        # coarse shape)

            voxel_style = data_dict['voxel_style']  # detailed shape
            Dmask_style = data_dict['Dmask_style']  # dilated mask on detailed shape (this is to be used in the
                                                    # discriminator so should be half detailed shape) uses pool and
                                                    # stride less than kernel to dilate it

            print(f"input shape {input.shape}")
            print(f"mask shape {mask.shape}")
            print(f"Dmask shape {Dmask.shape}")
            print(f"voxel_style shape {voxel_style.shape}")
            print(f"Dmask_style shape {Dmask_style.shape}")

            binvox_file = dset.files[i]
            name = os.path.dirname(binvox_file)

            voxel_centers_input = get_points_from_voxel(input)
            voxel_centers_input = normalize_vertices(voxel_centers_input)
            voxel_centers_input += np.array([-2, 0, 0])
            geometries.append(PointCloud(voxel_centers_input))
            box = get_unit_bbox((-2, 0, 0))
            geometries.append(box)
            voxel_centers_voxel_style = get_points_from_voxel(voxel_style)
            voxel_centers_voxel_style = normalize_vertices(voxel_centers_voxel_style)
            voxel_centers_voxel_style += np.array([-1, 0, 0])
            geometries.append(PointCloud(voxel_centers_voxel_style))
            box = get_unit_bbox((-1, 0, 0))
            geometries.append(box)

            tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
            voxel_centers_tmp_raw = get_points_from_voxel(tmp_raw)
            voxel_centers_tmp_raw = normalize_vertices(voxel_centers_tmp_raw)
            voxel_centers_tmp_raw += np.array([0, 0, 0])
            geometries.append(PointCloud(voxel_centers_tmp_raw))
            box = get_unit_bbox((0, 0, 0))
            geometries.append(box)

            xmin, xmax, ymin, ymax, zmin, zmax, pooled_tmp_raw = dset.get_voxel_bbox(tmp_raw, debug=True)
            cropped = dset.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)
            voxel_centers_cropped = get_points_from_voxel(cropped)
            voxel_centers_cropped = normalize_vertices(voxel_centers_cropped)
            voxel_centers_cropped += np.array([1, 0, 0])
            geometries.append(PointCloud(voxel_centers_cropped))
            box = get_unit_bbox((1, 0, 0))
            geometries.append(box)

            voxel_centers_mask = get_points_from_voxel(mask)
            voxel_centers_mask = normalize_vertices(voxel_centers_mask)
            voxel_centers_mask += np.array([2, 0, 0])
            geometries.append(PointCloud(voxel_centers_mask))
            box = get_unit_bbox((2, 0, 0))
            geometries.append(box)

            voxel_centers_Dmask = get_points_from_voxel(Dmask)
            voxel_centers_Dmask = normalize_vertices(voxel_centers_Dmask)
            voxel_centers_Dmask += np.array([3, 0, 0])
            geometries.append(PointCloud(voxel_centers_Dmask))
            box = get_unit_bbox((3, 0, 0))
            geometries.append(box)

            render_geometries(geometries, window_name=name, camera_json=CameraJsonPosition)

            # all_ones_tmp_raw = np.ones_like(tmp_raw)
            # all_ones_cropped = np.ones_like(cropped)
            # voxel_centers_pooled_tmp_raw = get_points_from_voxel(pooled_tmp_raw)
            # bound_points_pooled_tmp_raw = get_bound_points_from_voxel(pooled_tmp_raw)
            # reds = np.array([[255,0,0]]*len(bound_points_pooled_tmp_raw))
            # blues = np.array([[0,0,255]]*len(voxel_centers_pooled_tmp_raw))
            # interactive_plot([PointCloud(bound_points_pooled_tmp_raw, reds),
            #                   PointCloud(voxel_centers_pooled_tmp_raw, blues)])


            # genika to Dmask_style xrisimopoieite sto loss_d_real (me to discriminator output given voxel_style)
            # kai to Dmask (content) xrisimopoieite sto loss_d_fake i loss_g (me to discriminator output given fake)
            # kai to mask (content) xrisimopoieite sto generation given content kai style
            # opotan den me noiazei kanonika to height kai width tou kathe shape !!!

    def test_dataloader(self):

        def collate_fn(data):
            keys = [k for k in data[0]]
            new_data = {}
            for k in keys:
                new_data.setdefault(k, [])
                for d in data:
                    new_data[k].append(d[k])
            return new_data

        out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local/check_furniture_hu/i16o128_norm_and_align_random_rot"
        data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/preprocessed_data/furniture_hu_norm_and_align_random_rot"
        fpath = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/splits/furniture_hu_norm_and_align_random_rot/export.txt"
        config = dotdict({"asymmetry": True, 'gpu': 0, 'input_size': 32, 'output_size': 256, "filename": "model_filled.binvox"})
        dset = FlexDataset(data_dir, fpath, config, logger, filename=config['filename'])
        print(f"dset length: {len(dset)}")
        rand_loader = DataLoader(dataset=dset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        for data in rand_loader:
            print(data)
