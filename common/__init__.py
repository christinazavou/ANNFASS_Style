import numpy as np


_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float32).eps


class PointOutsideFaceException(Exception):
    def __init__(self, message):
        super().__init__(message)


class Vertices(list):
    def __init__(self):
        super().__init__()


class VertexNormals(list):
    def __init__(self):
        super().__init__()


class TextureCoords(list):
    def __init__(self):
        super().__init__()


class Face:

    def __init__(self, v_indices, vn_indices, vt_indices, material, component):
        self.v_indices = v_indices
        self.vn_indices = vn_indices
        self.vt_indices = vt_indices
        self.material = material
        self.component = component


class SampledPoint:

    def __init__(self, coords, normals, face_idx):
        self.coords = coords
        self.normals = normals
        self.face_idx = face_idx


class SampledPointWithColor:

    def __init__(self, sampled_point_coords, face_idx, uv, color):
        self.coords = sampled_point_coords
        self.face_idx = face_idx
        self.uv_interp = uv
        self.color_interp = color


class FullSampledPoint:

    def __init__(self, coords, normals, pca, curvature, face_idx):
        self.coords = coords
        self.normals = normals
        self.pca = pca
        self.curvature = curvature
        self.face_idx = face_idx

