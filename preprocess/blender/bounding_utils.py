import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from space_utils import max_distance_from_center_to_obj
from mesh_utils import get_centroid


def sphere_bound(objects):
    radius = 0.
    center = get_centroid(objects)
    for obj in objects:
        if obj.type == 'MESH':
            r = max_distance_from_center_to_obj(center, obj)
            if r > radius:
                radius = r
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center)
    bound_sphere = bpy.context.active_object
    return bound_sphere, radius


def get_bound_cube_coords(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def get_overall_bound_cube_corners(objects):
    bound_coords = []
    for obj in objects:
        if obj.type == 'MESH':
            bound_coords += get_bound_cube_coords(obj)
    bound_coords = np.array(bound_coords)
    xyz_min = np.amin(bound_coords, 0)
    xyz_max = np.amax(bound_coords, 0)
    return xyz_min, xyz_max


# def bound_an_object(obj, scene, name='3DBoundingBox'):
#     # note: this is not aligned with the axis; it is aligned with the object ;)
#     corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
#     faces_3Dbb = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7)]
#     mesh = bpy.data.meshes.new(name)
#     mesh.from_pydata(corners, [], faces_3Dbb)
#     newbbox = bpy.data.objects.new(name, mesh)
#     scene.collection.objects.link(newbbox)
#     return newbbox
