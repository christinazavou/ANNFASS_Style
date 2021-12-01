import json
import os
import sys

import bpy
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from common.utils import UndirectedGraph

from preprocess.blender.bounding_utils import get_overall_bound_cube_corners
from preprocess.blender.space_utils import vectors_distance, distance_of_objects
from preprocess.blender.colour_utils import DISTINCT_COLOURS_40, to_rgb
from preprocess.blender.camera_utils import get_or_add_camera
from preprocess.blender.renderer_utils import CyclesRenderer, EeveeRenderer
from preprocess.blender.io_utils import save_simple_mesh
from preprocess.blender.scene_utils import select_objects
from preprocess.blender.mesh_utils import objects_are_touching


def get_colors(groups):
    g_colours = {}
    idx_color = -1
    for gi, _ in enumerate(groups):
        idx_color += 1
        if idx_color == len(DISTINCT_COLOURS_40):
            idx_color = 0  # go over same colours
        g_colours[gi] = DISTINCT_COLOURS_40[idx_color]
    return g_colours


def save_colors(g_colours, g_colours_f):
    with open(g_colours_f, "w") as fout:
        json.dump(g_colours, fout, indent=2)


def load_colors(groups_f):
    with open(groups_f, "r") as fin:
        g_colours = json.load(fin)
        g_colours = {int(key): value for key, value in g_colours.items()}
    return g_colours


def get_materials_distinct_colors_40():
    material_colors = []
    for color_idx in range(len(DISTINCT_COLOURS_40)):
        mat = bpy.data.materials.new(name="Color{}".format(color_idx))
        mat.diffuse_color = to_rgb(DISTINCT_COLOURS_40[color_idx])
        material_colors.append(mat)
    return material_colors


def colorize(groups, g_colours, remove_material=True, all_same=False):
    material_colors = get_materials_distinct_colors_40()
    if remove_material:
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.data.materials.clear()
    for gi, group_names in enumerate(groups):
        idx_color = DISTINCT_COLOURS_40.index(g_colours[gi])
        mat = material_colors[idx_color] if not all_same else material_colors[0]
        for name in group_names:
            bpy.data.objects[name].active_material = mat


def save_groups(groups, groups_f):
    groups = {i: g for i, g in enumerate(groups)}
    with open(groups_f, "w") as fout:
        json.dump(groups, fout, indent=2)


def load_groups(groups_f):
    with open(groups_f, "r") as fin:
        groups = json.load(fin)
    print("loaded groups: ", groups)
    return groups


def visualize_groups(groups_dir, scene, on_gpu=True, transparent_back=False):
    camera = get_or_add_camera()
    # if on gypsum...Cycles renderer must be used .. however Eevees is preferable because cycles gives an artifact...
    if on_gpu:
        blender_renderer = CyclesRenderer(scene, transparent_background=transparent_back)
    else:
        blender_renderer = EeveeRenderer(scene, transparent_background=transparent_back)
    if transparent_back:
        blender_renderer.multi_view(bpy.data.objects, groups_dir, "tr_group_", camera)
    else:
        blender_renderer.multi_view(bpy.data.objects, groups_dir, "_grouped_", camera)


def save_ply_groups(groups, groups_dir, scene):
    ply_filenames = []
    for i, group in enumerate(groups):
        select_objects(scene, group)
        ply_filename = os.path.join(groups_dir, "group_{}.ply".format(i))
        save_simple_mesh(ply_filename)
        ply_filenames.append(ply_filename)
    return ply_filenames


class ComponentsGrouper:

    def __init__(self, logger, factor=1., debug_dir=None):
        self.factor = factor
        self.elements = None
        self.groups = None
        self.logger = logger
        self.threshold = None
        self.distances = None
        self.debug_dir = debug_dir

    def get_connectivity_threshold_of_style(self):
        assert self.elements != None
        elements_diagonals = []
        for obj_name in self.elements:
            xyz_min, xyz_max = get_overall_bound_cube_corners([bpy.data.objects[obj_name]])  # FIXME: check if bound cubes are always correct
            diagonal = vectors_distance(xyz_min, xyz_max)
            elements_diagonals.append(diagonal)
        self.threshold = np.mean(elements_diagonals) * self.factor
        self.logger.debug("diagonal threshold: {}".format(self.threshold))

    def _save_distances(self):
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)
            with open(os.path.join(self.debug_dir, "distance.csv"), "w") as fout:
                fout.write("name,"+",".join(self.elements)+"\n")
                for idx, name in enumerate(self.elements):
                    fout.write(name+","+",".join([str(d) for d in self.distances[idx]])+"\n")
            with open(os.path.join(self.debug_dir, "threshold.txt"), "w") as fout:
                fout.write(str(self.threshold))

    def add_all(self, elements):
        self.elements = elements
        self.get_connectivity_threshold_of_style()
        self.graph = UndirectedGraph(len(elements))
        self.distances = np.zeros((len(elements), len(elements)))

        for idx1 in range(0, len(elements) - 1):
            for idx2 in range(idx1 + 1, len(elements)):
                obj1 = bpy.data.objects[elements[idx1]]
                obj2 = bpy.data.objects[elements[idx2]]
                d = distance_of_objects(obj1, obj2)
                self.distances[idx1, idx2] = d
                if d <= self.threshold:
                    if d == 0:
                        self.logger.info(f"{obj1},{obj2} should be touching")
                    self.graph.add_edge(idx1, idx2)
                if objects_are_touching(obj1, obj2):  # not elif for debugging purposes
                    self.logger.info(f"{obj1},{obj2} touching")
                    self.graph.add_edge(idx1, idx2)  # no problem if same edge is added again

        self._save_distances()
        return self

    def group(self):
        assert self.elements != None
        groups = self.graph.connected_components()
        list_to_elements = lambda l: [self.elements[idx] for idx in l]
        self.groups = [list_to_elements(l) for l in groups]
        if len(self.groups) > 20:
            self.logger.warning("Number of groups > number of colours.")
        return self

    def get(self):
        return self.groups
