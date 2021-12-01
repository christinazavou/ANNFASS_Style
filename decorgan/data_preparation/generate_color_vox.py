import open3d as o3d
import numpy as np
import os

from common_utils import binvox_rw
from common_utils.utils import normalize_vertices, rotate_x, read_ply_v_c, get_points_from_voxel, \
    get_points_and_colors_from_voxel
from utils import get_vox_from_binvox_1over2
from utils.open3d_utils import PointCloud, get_unit_bbox

from common_utils.vox_model import Voxels


def get_512_colorvox_from_files(binvoxfile, plyfile, debug=False):

    with open(binvoxfile, "rb") as fin:
        vox = binvox_rw.read_as_3d_array(fin).data.astype(np.uint8)
    with open(binvoxfile, "rb") as fin:
        translation = np.array(binvox_rw.read_header(fin)[1])

    voxels = Voxels(vox, translation)

    vertices, colors = read_ply_v_c(plyfile)
    vertices = normalize_vertices(vertices)
    vertices = rotate_x(vertices, 90)

    if debug:
        reds = np.zeros_like(vertices)
        reds[:, 0] = 1
        blues = np.zeros_like(voxels.v00)
        blues[:, 2] = 1
        o3d.visualization.draw_geometries([PointCloud(voxels.v00, blues), PointCloud(vertices, reds)])

    voxels.update_colors(vertices, colors)
    # o3d.visualization.draw_geometries([PointCloud(voxels.vcenters, voxels.vcolors/255.)])

    voxel_model_512 = np.append(vox.reshape((512, 512, 512, 1)), voxels.colors, axis=3)
    return voxel_model_512


def get_colorvox(binvoxfile=None, plyfile=None, voxel_model_512=None):
    if voxel_model_512 is None:
        voxel_model_512 = get_512_colorvox_from_files(binvoxfile, plyfile)
    step_size = 2
    voxel_model_256 = voxel_model_512[0::step_size,0::step_size,0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_256 = np.maximum(voxel_model_256,voxel_model_512[i::step_size,j::step_size,k::step_size])
    return voxel_model_256


def get_colorvox_1over2(binvoxfile=None, plyfile=None, voxel_model_512=None):
    if voxel_model_512 is None:
        voxel_model_512 = get_512_colorvox_from_files(binvoxfile, plyfile)
    step_size = 4
    output_padding = 128-(256//step_size)
    voxel_model_128 = voxel_model_512[0::step_size,0::step_size,0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_128 = np.maximum(voxel_model_128,voxel_model_512[i::step_size,j::step_size,k::step_size])
    voxel_model_256 = np.zeros([256,256,256, 4],np.uint8)
    voxel_model_256[output_padding:-output_padding,
                    output_padding:-output_padding,
                    output_padding:-output_padding] = voxel_model_128
    return voxel_model_256


if __name__ == '__main__':
    binvox_folder = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components"
    ply_folder = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints_refinedTextures/groups_june17_colorPly_cut10.0K_pgc"
    for building in os.listdir(binvox_folder):
        print(f"building {building}")
        for group in os.listdir(os.path.join(binvox_folder, building)):
            print(f"group {group}")
            if not any(l in group for l in ["window", "door", "column", "tower", "dome"]):
                continue
            binvox_file = f"{binvox_folder}/{building}/{group}/model_filled.binvox"
            ply_file = f"{ply_folder}/{building}/{building}_{group.replace('style_mesh_', '')}.ply"
            if not os.path.exists(binvox_file) or not os.path.exists(ply_file):
                continue

            vox512 = get_512_colorvox_from_files(binvox_file, ply_file, debug=True)

            vox256 = get_colorvox(voxel_model_512=vox512)
            p, c = get_points_and_colors_from_voxel(vox256)
            o3d.visualization.draw_geometries([PointCloud(p, c/255.)], window_name=f"{building}/{group}")

            vox256 = get_colorvox(voxel_model_512=vox512)
            p, c = get_points_and_colors_from_voxel(vox256)
            o3d.visualization.draw_geometries([PointCloud(p, c/255.)], window_name=f"{building}/{group}")
