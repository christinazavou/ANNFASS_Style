import numpy as np
import os
from packaging import version

import open3d as o3d

from utils.io_helper import parse_simple_obj_file


def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def TriangleMesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def resample_mesh(mesh_cad, density=1):
    '''
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    '''
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    if np.min(faces) == 1:
        faces = faces - 1

    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                         vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc:acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (1 - np.sqrt(r[:, 0:1])) * A + \
        np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
        np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    return P


def test_1():
    directory = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/buildnet_component_refined/RELIGIOUSchurch_mesh2460"
    from utils.open3d_vis import render_obj
    w = 1024
    h = 768
    window_visible = False
    view_file = "ScreenCameraLocation.json"
    i = 0
    for component_dir in os.listdir(directory):
        i +=1
        obj_file = os.path.join(directory, component_dir, "model.obj")
        render_img = f"render{i}.png"
        render_obj(obj_file, w, h, window_visible, view_file, render_img, show=False)


def test_2():
    directory = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_component_v2_orient/RELIGIOUScathedral_mesh0986"
    from utils.open3d_vis import render_mesh
    w = 1024
    h = 768
    window_visible = False
    view_file = "ScreenCameraUnitCubeLocation.json"
    i = 0
    for component_dir in os.listdir(directory):
        i +=1
        obj_file = os.path.join(directory, component_dir, "model.obj")
        v, f = parse_simple_obj_file(obj_file)
        mesh = TriangleMesh(v, f)
        render_img = f"render{i}.png"
        render_mesh(mesh, w, h, window_visible, view_file, render_img, show=False)


def test_3():
    directory = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_component_v2_orient/RELIGIOUScathedral_mesh0986"
    i = 0
    for component_dir in os.listdir(directory):
        i +=1
        obj_file = os.path.join(directory, component_dir, "model.obj")
        v, f = parse_simple_obj_file(obj_file)
        mesh = TriangleMesh(v, f)
        o3d.visualization.draw_geometries([mesh],
                                          front=-np.array([-0.75, -0.5, 0.433]),
                                          lookat=np.array([0,0,0,]),
                                          up=np.array([-0.433, 0.86, 0.25]),
                                          zoom=0.8)


def get_unit_bbox(translate=(0, 0, 0), scale=1, color=(1, 0, 0)):
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    if version.parse(o3d.__version__) < version.parse("0.12.0.0"):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
    else:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
    if scale != 1:
        line_set.scale(scale)
    if translate != (0, 0, 0):
        line_set.translate(translate)
    if color != ():
        line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
    return line_set


if __name__ == '__main__':
    # test_1()
    test_2()
    # test_3()
