from preprocess.minkowski_encoding.utils import read_obj_with_components, read_plyfile, read_face_indices
import numpy as np


def test_read_obj_with_components_annfass():
    filename = 'resources/normalizedObj/16_English school/16_English school_01.obj'
    obj = read_obj_with_components(filename)
    vertices, faces, component_references, components = obj
    filename = 'resources/ply_7K/16_English school_01.ply'
    ply = read_plyfile(filename)
    filename = 'resources/faces_7K/16_English school_01.txt'
    face_indices = read_face_indices(filename)
    assert np.max(face_indices) <= faces.shape[0] - 1
    assert ply.shape[0] == face_indices.shape[0]
