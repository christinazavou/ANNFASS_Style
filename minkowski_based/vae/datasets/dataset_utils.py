import os
import collections

from scipy.linalg import expm, norm
from plyfile import PlyData
import open3d as o3d
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


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
    if np.min(faces) == 1:
        faces = faces - 1

    vertices = np.array(mesh_cad.vertices)

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


def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file)

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        assert lines[0] == 'OFF'

        parts = lines[1].split(' ')
        assert len(parts) == 3

        num_vertices = int(parts[0])
        assert num_vertices > 0

        num_faces = int(parts[1])
        assert num_faces > 0

        vertices = []
        for i in range(num_vertices):
            vertex = lines[2 + i].split(' ')
            vertex = [float(point) for point in vertex]
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split(' ')
            face = [int(index) for index in face]

            assert face[0] == len(face) - 1
            for index in face:
                assert index >= 0 and index < num_vertices

            assert len(face) > 1

            faces.append(face)

        return vertices, faces


def load_off(mesh_file, density, normalize=True):
    # Load a mesh, over sample, copy, rotate, voxelization
    assert os.path.exists(mesh_file), f"{mesh_file} doesnt exist"
    vertices, faces = read_off(mesh_file)
    vertices = np.array(vertices)
    faces = np.array(faces)
    assert faces.shape[1] == 4
    mesh = TriangleMesh(vertices, faces[:, 1:])
    if normalize:
        # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
        vmax = vertices.max(0, keepdims=True)
        vmin = vertices.min(0, keepdims=True)
        mesh.vertices = o3d.utility.Vector3dVector(
            (vertices - vmin) / (vmax - vmin).max())

    # Oversample points and copy
    xyz = resample_mesh(mesh, density=density)
    return xyz


def load_ply(ply_file, density, normalize=True):
    # Load a mesh, over sample, copy, rotate, voxelization
    assert os.path.exists(ply_file), f"{ply_file} doesn't exist"
    plydata = PlyData.read(ply_file)
    data = plydata.elements[0].data
    vertices = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T

    if normalize:
        # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
        vmax = vertices.max(0, keepdims=True)
        vmin = vertices.min(0, keepdims=True)
        vertices = (vertices - vmin) / (vmax - vmin)

    # Oversample points and copy
    xyz = vertices[np.random.choice(vertices.shape[0], density, False)]
    return xyz


def load_ply_mesh(mesh_file, density, resample=False, normalize=False):
    # Load a mesh, over sample, copy, rotate, voxelization
    assert os.path.exists(mesh_file), f"{mesh_file}  not found"
    plydata = PlyData.read(mesh_file)
    vertices = plydata['vertex'].data
    vertices = np.array([[v[0], v[1], v[2]] for v in vertices])

    if resample:
        faces = plydata['face'].data['vertex_indices']
        faces = np.array([[f[0], f[1], f[2]] for f in faces])

        mesh = TriangleMesh(vertices, faces)

        if normalize:
            # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            mesh.vertices = o3d.utility.Vector3dVector(
                (vertices - vmin) / (vmax - vmin).max())

        # Oversample points and copy
        xyz = resample_mesh(mesh, density=density)
        return xyz
    else:
        if normalize:
            vmax = vertices.max(0, axis=0)
            vmin = vertices.min(0, axis=0)
            vertices = (vertices - vmin) / (vmax - vmin).max()

        return vertices


def parse_simple_obj_file(obj_file):
    with open(obj_file, "r") as fin:
        lines = fin.readlines()
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(np.asarray(line[2:-1].split(" ")).astype(float))
        else:
            faces.append(np.asarray(line[2:-1].split(" ")).astype(float))
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if np.min(faces) == 1:
        faces = faces - 1
    return vertices, faces


def load_obj(mesh_file, density, normalize=True):
    # Load a mesh, over sample, copy, rotate, voxelization
    assert os.path.exists(mesh_file), f"{mesh_file} doesnt exist"
    vertices, faces = parse_simple_obj_file(mesh_file)
    vertices = np.array(vertices)
    faces = np.array(faces)
    assert faces.shape[1] == 3
    mesh = TriangleMesh(vertices, faces)
    if normalize:
        # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
        vmax = vertices.max(0, keepdims=True)
        vmin = vertices.min(0, keepdims=True)
        mesh.vertices = o3d.utility.Vector3dVector(
            (vertices - vmin) / (vmax - vmin).max())

    # Oversample points and copy
    xyz = resample_mesh(mesh, density=density)
    return xyz


def get_voxelized_data(xyz, resolution, transform=None):
    # Use color or other features if available
    feats = np.ones((len(xyz), 1))
    if transform:
        xyz, feats = transform(xyz, feats)
    # Get coords
    coords = xyz * resolution
    coords = np.floor(coords)
    # inds = ME.utils.sparse_quantize(coords, return_index=True)
    # return coords[inds], xyz[inds]
    coords, xyz = ME.utils.sparse_quantize(coords, xyz)
    return coords, xyz


# ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi, np.pi), (-np.pi / 64, np.pi / 64))
#
#
# # Rotation matrix along axis with angle theta
# def M(axis, theta):
#     return expm(np.cross(np.eye(3), axis / norm(axis) * theta))
#
#
# def get_transformation_matrix(use_augmentation=True,
#                               rotation_augmentation_bound=ROTATION_AUGMENTATION_BOUND,
#                               resolution=128):
#     voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
#
#     # Transform pointcloud coordinate to voxel coordinate.
#     # 1. Random rotation
#     rot_mat = np.eye(3)
#     if use_augmentation and rotation_augmentation_bound is not None:
#         if isinstance(rotation_augmentation_bound, collections.Iterable):
#             rot_mats = []
#             for axis_ind, rot_bound in enumerate(rotation_augmentation_bound):
#                 theta = 0
#                 axis = np.zeros(3)
#                 axis[axis_ind] = 1
#                 if rot_bound is not None:
#                     theta = np.random.uniform(*rot_bound)
#                 rot_mats.append(M(axis, theta))
#             # Use random order
#             np.random.shuffle(rot_mats)
#             rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
#         else:
#             raise ValueError()
#     rotation_matrix[:3, :3] = rot_mat
#     # 2. Scale and translate to the voxel space.
#     scale = 1.
#     if use_augmentation and resolution is not None:
#         scale = resolution  # todo: change it
#     np.fill_diagonal(voxelization_matrix[:3, :3], scale)
#     # Get final transformation matrix.
#     return voxelization_matrix, rotation_matrix
#
#
# def get_voxelize_data_with_features(coords, feats,
#                                     use_augmentation=True,
#                                     rotation_augmentation_bound=ROTATION_AUGMENTATION_BOUND,
#                                     resolution=128):
#     assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
#
#     # Get rotation and scale
#     M_v, M_r = get_transformation_matrix(use_augmentation=use_augmentation,
#                                          rotation_augmentation_bound=rotation_augmentation_bound,
#                                          resolution=resolution)
#     # Apply transformations
#     rigid_transformation = M_v
#     if use_augmentation:
#         rigid_transformation = M_r @ rigid_transformation
#
#     homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
#     coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])
#
#     coords_aug, feats = ME.utils.sparse_quantize(coords_aug, feats)
#
#     return coords_aug, feats, rigid_transformation.flatten()


def collate_pointcloud_with_features_fn(list_data):
    list_data = [l for l in list_data if l is not None]
    if len(list_data) == 0:
        return {}
    coords, feats, labels = list(zip(*list_data))

    sparse_coords, sparse_feats = ME.utils.sparse_collate(
        [torch.from_numpy(coord) for coord in coords],
        [torch.from_numpy(feat) for feat in feats])

    # Concatenate all lists
    return {
        'coords': sparse_coords,
        'feats': sparse_feats,
        'labels': torch.LongTensor(labels),
    }


def collate_content_style_cloud_with_features_fn(list_data):
    list_data = [l for l in list_data if l is not None]
    if len(list_data) == 0:
        return {}

    content_coords, content_feats, \
    content_detailed_coords, content_detailed_feats, \
    style_coords, style_feats, labels = list(zip(*list_data))

    assert len(content_coords) == len(content_feats) \
           == len(content_detailed_coords) == len(content_detailed_feats) \
           == len(style_coords) == len(style_feats)

    sparse_content_coords, sparse_content_feats = ME.utils.sparse_collate(
        [torch.from_numpy(coord) for coord in content_coords],
        [torch.from_numpy(feat) for feat in content_feats])

    sparse_content_detailed_coords, sparse_content_detailed_feats = ME.utils.sparse_collate(
        [torch.from_numpy(coord) for coord in content_detailed_coords],
        [torch.from_numpy(feat) for feat in content_detailed_feats])

    sparse_style_coords, sparse_style_feats = ME.utils.sparse_collate(
        [torch.from_numpy(coord) for coord in style_coords],
        [torch.from_numpy(feat) for feat in style_feats])

    # Concatenate all lists
    return {
        'content_coords': sparse_content_coords,
        'content_xyzs': sparse_content_feats,
        'content_detailed_coords': sparse_content_detailed_coords,
        'content_detailed_xyzs': sparse_content_detailed_feats,
        'style_coords': sparse_style_coords,
        'style_xyzs': sparse_style_feats,
        'labels': torch.LongTensor(labels),
    }


def collate_pointcloud_fn(list_data):
    list_data = [l for l in list_data if l is not None]
    if len(list_data) == 0:
        return {}
    coords, feats, labels = list(zip(*list_data))

    # Concatenate all lists
    return {
        'coords': ME.utils.batched_coordinates(coords),
        'xyzs': [torch.from_numpy(feat).float() for feat in feats],
        'labels': torch.LongTensor(labels),
    }


def collate_content_style_cloud_fn(list_data):
    list_data = [l for l in list_data if l is not None]
    if len(list_data) == 0:
        return {}

    content_coords, content_feats, \
    content_detailed_coords, content_detailed_feats, \
    style_coords, style_feats, labels = list(zip(*list_data))

    assert len(content_coords) == len(content_feats) \
           == len(content_detailed_coords) == len(content_detailed_feats) \
           == len(style_coords) == len(style_feats)

    # Concatenate all lists
    return {
        'content_coords': ME.utils.batched_coordinates(content_coords),
        'content_xyzs': [torch.from_numpy(feat).float() for feat in content_feats],
        'content_detailed_coords': ME.utils.batched_coordinates(content_detailed_coords),
        'content_detailed_xyzs': [torch.from_numpy(feat).float() for feat in content_detailed_feats],
        'style_coords': ME.utils.batched_coordinates(style_coords),
        'style_xyzs': [torch.from_numpy(feat).float() for feat in style_feats],
        'labels': torch.LongTensor(labels),
    }


def make_data_loader(dset, batch_size, shuffle, num_workers, repeat, collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': False,
        'drop_last': False
    }

    if repeat:
        args['sampler'] = InfSampler(dset, shuffle)
    else:
        args['shuffle'] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader
