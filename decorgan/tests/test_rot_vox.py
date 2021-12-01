import logging

import cv2
import torch
import math

import numpy as np
import mathutils
import matplotlib.pyplot as plt

from data_preparation.hu_yu_lun_buildnet.preprocess import load_obj, run_voxelize, write_obj, run_rendering
from runners.common import IM_AE_COMMON, IM_AE_STATIC
from utils import get_vox_from_binvox_1over2, get_voxel_bbox, voxel_renderer, get_torch_device, dotdict


class IM_AE(IM_AE_COMMON):

    def __init__(self, config):
        self.local = True
        self.real_size = 256
        self.mask_margin = 8
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.asymmetry = config.asymmetry
        self.sampling_threshold = 0.4
        self.render_view_id = 0
        if self.asymmetry: self.render_view_id = 6  # render side view for motorbike
        self.voxel_renderer = voxel_renderer(self.real_size)
        self.device = get_torch_device(config, logging.getLogger(__file__))
        self.upsample_rate = 4


def create_rotated_binvox():
    obj_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/preprocessed_data/data_with_rotations/Children_bed_1/model.obj"
    rot_obj_file = obj_file.replace(".obj", ".rot.obj")
    vox_file = obj_file.replace(".obj", ".binvox")
    rot_vox_file = rot_obj_file.replace(".obj", ".binvox")
    img_file = obj_file.replace(".obj", "myimg.jpg")
    rot_img_file = rot_obj_file.replace(".obj", "myimg.jpg")

    vertices, faces = load_obj(obj_file)
    angle = np.random.randint(0, 360)
    rot = mathutils.Matrix.Rotation(math.radians(angle), 3, (0, 1, 0))
    vertices_rot = np.matmul(vertices, rot)

    write_obj(rot_obj_file, vertices_rot, faces)

    run_voxelize(obj_file, vox_file, "/media/graphicslab/BigData/zavou/decor-gan-private/data_preparation/binvox")
    run_voxelize(rot_obj_file, rot_vox_file, "/media/graphicslab/BigData/zavou/decor-gan-private/data_preparation/binvox")

    run_rendering(vox_file)
    run_rendering(rot_vox_file)

    im_ae = IM_AE(config=dotdict({}))
    tmp_raw = get_vox_from_binvox_1over2(vox_file).astype(np.uint8)
    img = IM_AE_STATIC.vox_plot(tmp_raw, im_ae.local, im_ae.voxel_renderer, im_ae.sampling_threshold, im_ae.real_size)
    plt.figure()
    plt.imshow(img)
    plt.savefig(img_file)

    rot_tmp_raw = get_vox_from_binvox_1over2(rot_vox_file).astype(np.uint8)
    rot_img = IM_AE_STATIC.vox_plot(rot_tmp_raw, im_ae.local, im_ae.voxel_renderer, im_ae.sampling_threshold, im_ae.real_size)
    plt.figure()
    plt.imshow(rot_img)
    plt.savefig(rot_img_file)

    assert not np.array_equal(tmp_raw, rot_tmp_raw)


create_rotated_binvox()
