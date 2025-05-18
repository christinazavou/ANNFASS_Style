import math

from scipy.spatial.transform import Rotation as R

from common import _THRESHOLD_TOL_32, _THRESHOLD_TOL_64
from common.mesh_utils_io import *

try:
    from common.mesh_utils_open3d import *
except Exception as e:
    print(f"WARNING: Could not load mesh utils using open3d:\n{e}")


def get_obj_centroid(vertices):
    assert isinstance(vertices, np.ndarray)
    min_xyz = np.amin(vertices, axis=0)
    max_xyz = np.amax(vertices, axis=0)
    mid_xyz = (max_xyz + min_xyz) / 2.
    return mid_xyz


def centralize_obj(vertices):
    centroid = get_obj_centroid(vertices)
    vertices -= centroid
    return vertices


def rotation_matrix_4x4(angle_degrees, axis):
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle_degrees)
    # Make sure axis is a numpy array and normalized
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    # Create a Rotation object from rotation vector (axis * angle)
    rot = R.from_rotvec(angle_rad * axis)
    # Get 3x3 rotation matrix
    rot_mat_3x3 = rot.as_matrix()
    # Create 4x4 identity matrix
    rot_mat_4x4 = np.eye(4)
    # Put the 3x3 rotation matrix in the top-left corner
    rot_mat_4x4[:3, :3] = rot_mat_3x3
    return rot_mat_4x4


def rotate(vertices, angle=30, axis=(0, 1, 0)):
    assert isinstance(vertices, np.ndarray)
    tr = rotation_matrix_4x4(angle, axis)
    vertices_tr = np.c_[vertices, np.ones(vertices.shape[0])] @ tr
    vertices = vertices_tr[:, 0:3]
    return vertices


def scale_in_each_direction(vertex, scale_factor):
    scale_xyz = np.array(([[scale_factor, 0, 0],
                           [0, scale_factor, 0],
                           [0, 0, scale_factor]]))
    return vertex @ scale_xyz


def centralize_and_unit_scale(obj_fn_in, obj_fn_out, centroid=np.array([0., 0, 0]), scale=1.):
    with open(obj_fn_in, "r") as fin, open(obj_fn_out, "w") as fout:
        for line in fin.readlines():
            if line.startswith("v "):
                vertex = line.strip().split()
                vertex = np.array([float(vertex[1]), float(vertex[2]), float(vertex[3])])
                vertex -= centroid
                vertex = scale_in_each_direction(vertex, scale)
                fout.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))
            else:
                fout.write(line)


def get_component_mesh(obj: ObjMeshWithComponents, component_id: str, return_map=False):
    faces_indices = np.array([f.v_indices for f in obj.faces if f.component == component_id])
    new_idx_coords = []
    new_coords_len = 0
    old_idx_new_idx = dict()
    new_faces = []
    for face_indices in faces_indices:
        new_face = []
        for v_idx in face_indices:
            if v_idx not in old_idx_new_idx:
                new_idx_coords.append(obj.vertex_coords[v_idx])
                old_idx_new_idx[v_idx] = new_coords_len
                new_coords_len += 1
            new_face.append(old_idx_new_idx[v_idx])
        new_faces.append(new_face)
    vertices = np.array(new_idx_coords)
    if return_map:
        return vertices, new_faces, old_idx_new_idx
    return vertices, new_faces


def get_component_points(obj: ObjMeshWithComponents,
                         points: Union[SampledPoints, SampledPointWithColor],
                         component_id: str):
    return [point for point in points if obj.faces[point.face_idx].component == component_id]


def get_component_vertices(obj: ObjMeshWithComponents,
                           points: Union[SampledPoints, SampledPointWithColor],
                           component_id: str):
    return [point.coords for point in points if obj.faces[point.face_idx].component == component_id]


def normalize_coords(coords, method="sphere"):
    """ Normalize coordinates """
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    if method.lower() == "sphere":
        radius = bounding_sphere_radius(coords=centered_coords)
    elif method.lower() == "box":
        radius = bounding_box_diagonal(coords=centered_coords)
    else:
        print("Unknown normalization method {}".format(method))
        exit(-1)

    radius = np.maximum(radius, _THRESHOLD_TOL_64 if radius.dtype == np.float64 else _THRESHOLD_TOL_32)

    return centered_coords / radius


def bounding_box_diagonal(coords):
    """ Return bounding box diagonal """
    bb_diagonal = np.array([np.amax(coords[:, 0]), np.amax(coords[:, 1]), np.amax(coords[:, 2])]) - \
                  np.array([np.amin(coords[:, 0]), np.amin(coords[:, 1]), np.amin(coords[:, 2])])
    bb_diagonal_length = np.sqrt(np.sum(bb_diagonal ** 2))

    return bb_diagonal_length


def bounding_sphere_radius(coords):
    """ Return bounding sphere radius """
    radius = np.max(np.sqrt(np.sum(coords ** 2, axis=1)))

    return radius


def remove_isolated_vertices(vertices: np.ndarray, triangles: np.ndarray):
    assert vertices.dtype == float and triangles.dtype == int

    vertices_mapping = np.full([len(vertices)], -1, np.int32)
    for i in range(len(triangles)):
        for j in range(3):
            vertices_mapping[triangles[i, j]] = 1
    counter = 0
    for i in range(len(vertices)):
        if vertices_mapping[i] > 0:
            vertices_mapping[i] = counter
            counter += 1

    vertices = vertices[vertices_mapping >= 0]
    triangles = vertices_mapping[triangles]

    return vertices, triangles


if __name__ == '__main__':
    # myobj = ObjMeshBasic("/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_semifinal/normalizedObj/01_Cathedral_of_Holy_Wisdom/01_Cathedral_of_Holy_Wisdom.obj")
    # print(myobj)
    # myobj = ObjMeshWithComponents("/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_semifinal/normalizedObj/01_Cathedral_of_Holy_Wisdom/01_Cathedral_of_Holy_Wisdom.obj")
    # print(myobj)
    # myobj = ObjMeshWithComponentsAndMaterials(
    #     "/home/graphicslab/Desktop/objchec/02_Panagia_Chrysaliniotissa/02_Panagia_Chrysaliniotissa.obj")
    # print(myobj)
    # from utils import STYLES
    # export_selection_obj(
    #     "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASSSamplePoints/normalizedObj/23_Cyprus_Archaeological_Museum_portico/23_Cyprus_Archaeological_Museum_portico.obj",
    #     "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASSSamplePoints/normalizedObj/23_Cyprus_Archaeological_Museum_portico/23_Cyprus_Archaeological_Museum_portico_style_mesh.obj",
    #     STYLES)
    # obj = ObjMeshWithAdjacency("/home/christina/Documents/annfass_playground/colour_point_cloud/full_obj/COMMERCIALcastle_mesh0365.obj")
    # ply = PlyWithAdjacency("/home/christina/Documents/annfass_playground/triangulated_meshes/ply/COMMERCIALcastle_mesh0904.ply")
    from utils import STYLES
    export_selection_obj("/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj/RELIGIOUScastle_mesh0349/RELIGIOUScastle_mesh0349.obj",
                         "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj/RELIGIOUScastle_mesh0349/RELIGIOUScastle_mesh0349_style_mesh.obj",
                         STYLES)
    # obj = ObjMeshWithComponentsAndMaterials("/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_april/normalizedObj/01_Cathedral_of_Holy_Wisdom/01_Cathedral_of_Holy_Wisdom.obj")
    # print(obj)
    # mtl = MTLMaterials(
    #     "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_manual/normalizedObj/23_Cyprus_Archaeological_Museum_portico/23_Cyprus_Archaeological_Museum_portico.mtl")
    # print(mtl)
    # v, f = read_ply_data("/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/uncom_mesh/RELIGIOUSschool_building_mesh0855/10_10_parapet_merlon__unknown.ply")
    # print(len(v), len(f))
