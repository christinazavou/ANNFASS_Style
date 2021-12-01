import json
import os

import numpy as np
import pandas as pd
from plyfile import PlyData


def fix_component(component):
    return component.split(".")[0]


def fix_groups(groups_file, fixed_groups_file):
    with open(groups_file, "r") as fin:
        groups = json.load(fin)
    for group_name, group_components in groups.items():
        group_components = [name.split(".")[0] for name in group_components]
        groups[group_name] = group_components
    with open(fixed_groups_file, "w") as fout:
        json.dump(groups, fout, indent=4)


def read_face_indices(filename, cut_at=-1):
    if cut_at == -1:
        return np.loadtxt(filename).astype(np.int32)
    else:
        assert cut_at > 0
        return np.loadtxt(filename).astype(np.int32)[:cut_at]


def read_plyfile(filepath, verbose=False):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        df = pd.DataFrame(plydata.elements[0].data)
        if verbose:
            print("ply data: ", list(df.columns))
        return df.values


def read_obj(obj_fn):
    """
        Read obj
    :param obj_fn: str
    :return:
        vertices: N x 3, numpy.ndarray(float)
    faces: M x 3, numpy.ndarray(int)
    component_references: M x 1, numpy.ndarray(int) this is a reference to the component index for each face
    """

    assert (os.path.isfile(obj_fn))

    # Return variables
    vertices, faces, component_references = [], [], []

    with open(obj_fn, 'r') as f_obj:
        component = -1
        # Read obj geometry
        for line in f_obj:
            line = line.strip().split(' ')
            if line[0] == 'v':
                # Vertex row
                assert (len(line) == 4)
                vertex = [float(line[1]), float(line[2]), float(line[3])]
                vertices.append(vertex)
            if line[0] == 'o':
                # object row
                assert len(line) in [1, 2]
                component += 1
            if line[0] == 'f':
                # Face row
                face = [float(line[1].split('/')[0]), float(line[2].split('/')[0]), float(line[3].split('/')[0])]
                faces.append(face)
                component_references.append(component)

    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    component_references = np.vstack(component_references)

    assert faces.shape[0] == component_references.shape[0]
    return vertices, faces.astype(np.int32), component_references.astype(np.int32)


def read_obj_with_components(obj_fn):
    """
        Read obj
    :param obj_fn: str
    :return:
        vertices: N x 3, numpy.ndarray(float)
    faces: M x 3, numpy.ndarray(int)
    component_references: M x 1, numpy.ndarray(int) this is a reference to the component index for each face
    """

    assert (os.path.isfile(obj_fn))

    # Return variables
    vertices, faces, component_references, components = [], [], [], []

    with open(obj_fn, 'r') as f_obj:
        component = -1
        # Read obj geometry
        for line in f_obj:
            line = line.strip().split(' ')
            if line[0] == 'v':
                # Vertex row
                assert (len(line) == 4)
                vertex = [float(line[1]), float(line[2]), float(line[3])]
                vertices.append(vertex)
            if line[0] == 'o':
                # object row
                assert len(line) == 2
                component += 1
                components.append(line[1])
            if line[0] == 'f':
                # Face row
                face = [float(line[1].split('/')[0]), float(line[2].split('/')[0]), float(line[3].split('/')[0])]
                faces.append(face)
                component_references.append(component)

    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    component_references = np.vstack(component_references)
    components = np.vstack(components)

    assert faces.shape[0] == component_references.shape[0]
    return vertices, faces.astype(np.int32), component_references.astype(np.int32), components
