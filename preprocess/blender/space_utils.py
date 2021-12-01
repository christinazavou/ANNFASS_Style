import math

import numpy as np


def rectangle_diagonal(dim1, dim2):
    return math.sqrt(dim1**2 + dim2**2)


def cube_diagonal(dim1, dim2, dim3):
    return math.sqrt(dim1**2 + dim2**2 + dim3**2)


def vectors_distance(vec0, vec1):
    return np.linalg.norm(vec0 - vec1)


def vector_length(vec):
    return np.linalg.norm(vec)


# def max_distance_from_center_to_obj(center, obj):
#     maxdis = 0
#     for corner in obj.bound_box:
#         dis = vectors_distance(center, obj.matrix_world @ Vector(corner))
#         if maxdis < dis:
#             maxdis = dis
#     return maxdis
def max_distance_from_center_to_obj(center, obj):
    maxdis = 0
    for vertex in obj.data.vertices:
        dis = vectors_distance(center, obj.matrix_world @ vertex.co)
        if maxdis < dis:
            maxdis = dis
    return maxdis


def get_translation_around_point(moving_location, center_point_location, degrees=30):
    angle = math.radians(degrees)
    x = math.cos(angle) * (moving_location.x - center_point_location.x) - \
        math.sin(angle) * (moving_location.y - center_point_location.y)
    y = math.sin(angle) * (moving_location.x - center_point_location.x) + \
        math.cos(angle) * (moving_location.y - center_point_location.y)
    return x, y, 0


def get_orbit_location(camera_location, center_point_location, degrees):
    tx, ty, tz = get_translation_around_point(camera_location, center_point_location, degrees)
    return center_point_location.x + tx, center_point_location.y + ty, camera_location.z


# def distance_of_objects(obj1, obj2):  # FIXME: check whether sometimes bound_box is incorrect
#     distances = []
#     for corner1 in obj1.bound_box:
#         for corner2 in obj2.bound_box:
#             dis = vectors_distance(obj1.matrix_world @ Vector(corner1),
#                                    obj2.matrix_world @ Vector(corner2))
#             distances.append(dis)
#     return min(distances)
def distance_of_objects(obj1, obj2):
    coords1 = [(obj1.matrix_world @ v.co) for v in obj1.data.vertices]
    coords2 = [(obj2.matrix_world @ v.co) for v in obj2.data.vertices]
    distances = np.sum(np.abs(np.array(coords1)[:, None, :] - np.array(coords2)[None, :, :]), axis=-1)
    return distances.min()


def vector_is_facing_upwards_or_downwards(vec, msg=None, logger=None):
    upwards_vec = np.array([0, 0, 1])
    vec_cp = np.array([vec.x, vec.y, vec.z])
    d = np.dot(vec_cp, upwards_vec) / (vector_length(vec_cp)*1.0)
    assert (-1 <= d <= 1), "OPS: {}, msg: {}".format(vec_cp, msg)
    angle = math.degrees(math.acos(d))
    assert angle <= 180, "How comes that angle is {}".format(angle)
    if angle <= 40:
        if logger:
            logger.debug("facing upwards")
        return True
    if angle >= 140:
        if logger:
            logger.debug("facing downwards")
        return True
    return False

