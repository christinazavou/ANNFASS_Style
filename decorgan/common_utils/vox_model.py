import numpy as np
from scipy import spatial

from common_utils.utils import get_points_from_voxel


def get_start_points_from_voxel(vox_model, threshold=0):
    xs, ys, zs = np.where(vox_model > threshold)
    normalized_start_point_xs = np.arange(vox_model.shape[0]) / vox_model.shape[0]
    normalized_start_point_ys = np.arange(vox_model.shape[1]) / vox_model.shape[1]
    normalized_start_point_zs = np.arange(vox_model.shape[2]) / vox_model.shape[2]
    xs_full = normalized_start_point_xs[xs]
    ys_full = normalized_start_point_ys[ys]
    zs_full = normalized_start_point_zs[zs]
    points = np.vstack([xs_full, ys_full, zs_full]).T
    return points, xs, ys, zs


class Voxels:

    def __init__(self, vox: np.ndarray, translation: np.ndarray = None):
        self.vox = vox
        self.colors = np.zeros(vox.shape+(3,)).astype(np.int8)
        self.vox_size = 1. / np.array(vox.shape)
        self.translation = translation
        v00, self.xs_indices, self.ys_indices, self.zs_indices = get_start_points_from_voxel(self.vox)
        self.v00 = v00 + self.translation
        vcenters = get_points_from_voxel(self.vox)
        self.vcenters = vcenters + self.translation
        self.vcolors = np.zeros(self.vcenters.shape).astype(np.int8)

    def update_color(self, point, color):
        v11 = self.v00 + self.vox_size

        indices = np.where(np.logical_and(np.all(self.v00 <= point, 1), np.all(point <= v11, 1)))[0]
        for idx in indices:
            if np.all(self.colors[self.xs_indices[idx], self.ys_indices[idx], self.zs_indices[idx]] == 0):
                self.colors[self.xs_indices[idx],
                            self.ys_indices[idx],
                            self.zs_indices[idx]] = color
            else:                self.colors[self.xs_indices[idx],
                self.ys_indices[idx],
                self.zs_indices[idx]] = (self.colors[self.xs_indices[idx],
                                                     self.ys_indices[idx],
                                                     self.zs_indices[idx]] + color) / 2

    def update_colors(self, points, colors):
        tree = spatial.KDTree(points, leafsize=500)
        distances, indices = tree.query(self.vcenters)
        self.vcolors = colors[indices]
        self.colors[self.xs_indices, self.ys_indices, self.zs_indices] = self.vcolors

    def get_colors(self,):
        return self.colors[self.xs_indices, self.ys_indices, self.zs_indices]
