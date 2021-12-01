import numpy as np
import torch
import torch.nn as nn
from scipy import spatial
from torch import Tensor

from common.chamfer_pytorch import CustomChamferDistance
from common.mesh_utils import rotate, centralize_obj, write_ply


def chamfer_distance_numpy(point_set_a, point_set_b, bidirectional=False):
    assert point_set_a.shape[0] == point_set_b.shape[0]
    assert point_set_a.shape[2] == point_set_b.shape[2]

    # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
    # dimension D).
    difference = (
        np.expand_dims(point_set_a, axis=-2) -
        np.expand_dims(point_set_b, axis=-3))
    # Calculate the square distances between each two points: |ai - bj|^2.
    square_distances = np.einsum("...i,...i->...", difference, difference)

    minimum_square_distance_a_to_b = np.min(square_distances, axis=-1)

    if bidirectional:
        minimum_square_distance_b_to_a = np.min(square_distances, axis=-2)
        return np.mean(minimum_square_distance_a_to_b, axis=-1) + np.mean(minimum_square_distance_b_to_a, axis=-1)
    else:
        return np.mean(minimum_square_distance_a_to_b, axis=-1)


def chamfer_distance_sklearn(array1, array2):
    num_point1 = array1.shape[0]
    num_point2 = array2.shape[0]
    tree1 = spatial.KDTree(array1, leafsize=num_point1)  # or cKDTree
    tree2 = spatial.KDTree(array2, leafsize=num_point2)  # or cKDTree
    distances1, _ = tree1.query(array2)
    distances2, _ = tree2.query(array1)
    av_dist1 = np.mean(distances1)
    av_dist2 = np.mean(distances2)
    return av_dist1 + av_dist2


class ChamferDistance:
    def __call__(self, x, y):

        """Chamfer distance between two point clouds
        Parameters
        ----------
        x: numpy array [n_points_x, n_dims]
            first point cloud
        y: numpy array [n_points_y, n_dims]
            second point cloud
        Returns
        -------
        chamfer_dist: float
            bidirectional Chamfer distance
        """

        return chamfer_distance_numpy(np.expand_dims(x, axis=-3), np.expand_dims(y, axis=-3))[0]


class ChamferPytorch(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(ChamferPytorch, self).__init__()
        self.chamferDist = CustomChamferDistance()
        self.device = device

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_3dim = x.unsqueeze(0)
        y_3dim = y.unsqueeze(0)

        loss = self.chamferDist(x_3dim, y_3dim, bidirectional=True, reduction="none")
        loss = loss.mean()
        return loss


def get_min_chamfer_pytorch(point_cloud1, point_cloud2, angle=10, axis=(0, 1, 0), filepath=None, stop_at=None):
    # centralize components
    point_cloud1 = centralize_obj(point_cloud1)
    point_cloud2 = centralize_obj(point_cloud2)

    min_chamfer = ChamferPytorch()(torch.tensor(point_cloud1).cuda(), torch.tensor(point_cloud2).cuda())
    if filepath:
        write_ply(filepath.replace(".ply", "init.ply"), point_cloud2)
    if stop_at is not None and min_chamfer <= stop_at:
        return min_chamfer

    # rotate one component
    for idx in range(360 // angle):
        point_cloud2 = rotate(point_cloud2, angle=angle, axis=axis)
        if filepath:
            write_ply(filepath.replace(".ply", f"{idx}.ply"), point_cloud2)
        chamf_dist = ChamferPytorch()(torch.tensor(point_cloud1).cuda(), torch.tensor(point_cloud2).cuda())
        if chamf_dist < min_chamfer:
            min_chamfer = chamf_dist
        if stop_at is not None and min_chamfer <= stop_at:
            return min_chamfer
    return min_chamfer
