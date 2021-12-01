from plyfile import PlyData, PlyElement
import numpy as np
import math


def read_ply_v_c(ply_file):
    plydata = PlyData.read(ply_file)
    vertices = plydata.elements[0].data
    colors = np.array([vertices['red'], vertices['green'], vertices['blue']], dtype=np.int32).T
    vertices = np.array([vertices['x'], vertices['y'], vertices['z']], dtype=np.float32).T
    return vertices, colors


def normalize_vertices(vertices):
    # normalize diagonal=1
    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])
    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)

    vertices[:, 0] = (vertices[:, 0] - x_mid) / scale
    vertices[:, 1] = (vertices[:, 1] - y_mid) / scale
    vertices[:, 2] = (vertices[:, 2] - z_mid) / scale
    return vertices


def rotate_y(vertices, angle):
    angle = math.radians(angle)
    mat = np.array([
        [math.cos(angle), 0,  math.sin(angle)],
        [0,                              1, 0],
        [-math.sin(angle), 0, math.cos(angle)],
    ])
    return vertices @ mat


def rotate_z(vertices, angle):
    angle = math.radians(angle)
    mat = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0],
        [0,                              0, 1],
    ])
    return vertices @ mat


def rotate_x(vertices, angle):
    angle = math.radians(angle)
    mat = np.array([
        [1,                              0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle),  math.cos(angle)],
    ])
    return vertices @ mat


def get_points_from_voxel(vox_model, threshold=0):
    xp, yp, zp = np.where(vox_model > threshold)
    normalized_mid_point_xs = np.linspace(0, 1, vox_model.shape[0]) + 1 / (vox_model.shape[0] * 2)
    normalized_mid_point_ys = np.linspace(0, 1, vox_model.shape[1]) + 1 / (vox_model.shape[1] * 2)
    normalized_mid_point_zs = np.linspace(0, 1, vox_model.shape[2]) + 1 / (vox_model.shape[2] * 2)
    # mid_point_xs = np.arange(vox_model.shape[0])
    # mid_point_ys = np.arange(vox_model.shape[1])
    # mid_point_zs = np.arange(vox_model.shape[2])
    xp = normalized_mid_point_xs[xp]
    yp = normalized_mid_point_ys[yp]
    zp = normalized_mid_point_zs[zp]
    points = np.vstack([xp, yp, zp]).T
    return points


def get_points_and_colors_from_voxel(vox_model, threshold=0):
    xp, yp, zp = np.where(vox_model[:,:,:,0] > threshold)

    colors = vox_model[xp, yp, zp, 1:]

    normalized_mid_point_xs = np.linspace(0, 1, vox_model.shape[0]) + 1 / (vox_model.shape[0] * 2)
    normalized_mid_point_ys = np.linspace(0, 1, vox_model.shape[1]) + 1 / (vox_model.shape[1] * 2)
    normalized_mid_point_zs = np.linspace(0, 1, vox_model.shape[2]) + 1 / (vox_model.shape[2] * 2)
    # mid_point_xs = np.arange(vox_model.shape[0])
    # mid_point_ys = np.arange(vox_model.shape[1])
    # mid_point_zs = np.arange(vox_model.shape[2])
    xp = normalized_mid_point_xs[xp]
    yp = normalized_mid_point_ys[yp]
    zp = normalized_mid_point_zs[zp]
    points = np.vstack([xp, yp, zp]).T
    return points, colors


def parse_buildings_csv(filename):
    buildings = []
    with open(filename, "r") as f:
        for line in f:
            buildings.append(line.strip().split(";")[1])
    print("buildings to process: {}".format(buildings))
    return buildings

