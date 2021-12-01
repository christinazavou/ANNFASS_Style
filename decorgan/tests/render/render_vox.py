import binvox_rw
import numpy as np
import mcubes

from runners.common import IM_AE_STATIC
from utils.open3d_render import render_geometries
from utils.open3d_utils import TriangleMesh, get_unit_bbox
from utils import normalize_vertices, get_voxel_bbox, get_vox_from_binvox_1over2, crop_voxel, recover_voxel, \
    CameraJsonPosition, get_vox_from_binvox
import open3d as o3d
import pickle
import torch
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2


# obj_model_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/RELIGIOUSchurch_mesh3135/style_mesh_group44_129_129_door__unknown/model.obj"
# voxel_model_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/RELIGIOUSchurch_mesh3135/style_mesh_group44_129_129_door__unknown/model_filled.binvox"
# with open(voxel_model_file, 'rb') as fin:
#     voxel_model_512 = binvox_rw.read_as_3d_array(fin, fix_coords=True).data.astype(np.uint8)

tsdf_model_file = "../RELIGIOUSchurch_mesh3135_style_mesh_group44_129_129_door__unknown.off.pkl"
with open(tsdf_model_file, "rb") as fin:
    voxel_model_512 = pickle.load(fin)

# import trimesh
# m = trimesh.load(obj_model_file)
# voxel_model_512 = m.voxelized(pitch=1.0/512).matrix.astype(np.int)

v, f = mcubes.marching_cubes(voxel_model_512, 0)
v = normalize_vertices(v)
m = TriangleMesh(v, f)
o3d.visualization.draw_geometries([m, get_unit_bbox((-0.5, -0.5, -0.5))], window_name='init')
exit()

# tmp_raw = get_vox_from_binvox_1over2(voxel_model_file)
# v, f = mcubes.marching_cubes(tmp_raw, 0)
# v = normalize_vertices(v)
# m = TriangleMesh(v, f)
# o3d.visualization.draw_geometries([m, get_unit_bbox((-0.5, -0.5, -0.5))], window_name='tmp_raw 1over2')
#
# tmp_raw = get_vox_from_binvox(voxel_model_file)
# v, f = mcubes.marching_cubes(tmp_raw, 0)
# v = normalize_vertices(v)
# m = TriangleMesh(v, f)
# o3d.visualization.draw_geometries([m, get_unit_bbox((-0.5, -0.5, -0.5))], window_name='tmp_raw')


import torch.nn.functional as F
r = F.max_pool3d(torch.from_numpy(voxel_model_512).unsqueeze(0).float(),
                 kernel_size=2, stride=2, padding=0).detach().numpy()[0]
v, f = mcubes.marching_cubes(r, 0)
v = normalize_vertices(v)
m = TriangleMesh(v, f)
o3d.visualization.draw_geometries([m, get_unit_bbox((-0.5, -0.5, -0.5))], window_name='r')



voxel_model_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/RELIGIOUSchurch_mesh3135/style_mesh_group44_129_129_door__unknown/model_filled_christina.binvox"

with open(voxel_model_file, 'rb') as fin:
    voxel_model_256 = binvox_rw.read_as_3d_array(fin, fix_coords=True).data.astype(np.uint8)

v, f = mcubes.marching_cubes(voxel_model_256, 0)
v = normalize_vertices(v)
m = TriangleMesh(v, f)
o3d.visualization.draw_geometries([m, get_unit_bbox((-0.5, -0.5, -0.5))], window_name='init_christina')

