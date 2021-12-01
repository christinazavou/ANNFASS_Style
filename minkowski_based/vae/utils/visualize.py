import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from datasets.dataset_utils import PointCloud


def visualize_result(inp_coords, out_coords, resolution, window_name='Open3D', width=1920, height=1080):
    pcd = PointCloud(out_coords)
    o3d.estimate_normals(pcd)
    pcd.translate([0.6 * resolution, 0, 0])
    # pcd.rotate(M)
    opcd = PointCloud(inp_coords)
    opcd.translate([-0.6 * resolution, 0, 0])
    o3d.estimate_normals(opcd)
    # opcd.rotate(M)
    o3d.visualization.draw_geometries([pcd, opcd], window_name=window_name, width=width, height=height)


def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10,
                        azim=240, axis=None, title=None, xyzlim = (-0.5, 0.5), *args, **kwargs):
    # plt.switch_backend('agg')
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(*xyzlim)
        ax.set_ylim3d(*xyzlim)
        ax.set_zlim3d(*xyzlim)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig
