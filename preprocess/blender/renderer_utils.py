import os
import sys

import bpy

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from scene_utils import set_active_scene, remove_obj
from bounding_utils import sphere_bound
from mesh_utils import get_centroid
from camera_utils import set_camera_view_on_bbox_sphere, rotate_towards_target
from space_utils import get_orbit_location
from colour_utils import DAYLIGHT_COLOR, SUNLIGHT_COLOR


class Renderer:

    @staticmethod
    def set_backface_culling(scene):
        print("Setting backface culling")
        for obj in scene.objects:
            if obj.type == 'MESH':
                obj.active_material.use_backface_culling = True

    def __init__(self, scene, low_resolution=True, fast_and_cheap=True, transparent_background=False):
        self.scene = scene
        # self.scene.render.image_settings.file_format = 'JPEG'
        if low_resolution:
            self.set_low_resolution()
        else:
            self.set_medium_resolution()
        if fast_and_cheap:
            self.set_fast_and_cheap_performance()
        if transparent_background:
            self.scene.render.film_transparent = True
        self.scene.render.image_settings.color_mode = 'RGBA'

    def __call__(self, filepath):
        assert '.jpg' in filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.scene.render.filepath = filepath
        set_active_scene(self.scene.name)
        bpy.ops.render.render(write_still=True)

    def set_fast_and_cheap_performance(self):
        self.scene.render.tile_y = 128
        self.scene.render.tile_x = 128

    def set_low_resolution(self):
        self.scene.render.image_settings.quality = 65
        self.scene.render.resolution_x = 512
        self.scene.render.resolution_y = 512

    def set_medium_resolution(self):
        self.scene.render.image_settings.quality = 80
        self.scene.render.resolution_x = 1024
        self.scene.render.resolution_y = 1024

    def multi_view(self, objects, render_dir, base_name, camera):
        bbox, radius = sphere_bound(objects)
        center = get_centroid(objects)
        set_camera_view_on_bbox_sphere(camera, bbox, radius, center)
        remove_obj(bbox)

        self(os.path.join(render_dir, base_name + ".jpg"))
        angle_rotations = 6
        for angle_rotation in range(angle_rotations-1):
            camera.location = get_orbit_location(camera.location, center, 360//angle_rotations)
            rotate_towards_target(camera, center)
            self(os.path.join(render_dir, base_name+"r{}.jpg".format(angle_rotation)))


class FreestyleRenderer(Renderer):
    def __init__(self, scene, low_resolution=True, fast_and_cheap=True, transparent_background=False):
        super().__init__(scene, low_resolution, fast_and_cheap, transparent_background)
        self._set_freestyle()

    def _clean_node_tree_links(self):
        tree = self.scene.node_tree
        for link in tree.links:
            tree.links.remove(link)

    def _get_node_tree_link(self, from_n, to_n):
        tree = self.scene.node_tree
        for link in tree.links:
            if link.from_node.name == from_n:
                if link.to_node.name == to_n:
                    return link

    def _get_or_add_alphaover_node(self):
        tree = self.scene.node_tree
        for node in tree.nodes:
            if node.name == "CompositorNodeAlphaOver":
                return node
        alphaover = tree.nodes.new("CompositorNodeAlphaOver")
        alphaover.inputs[1].default_value = (1, 1, 1, 1)
        return alphaover

    def _set_freestyle(self):
        self.scene.render.use_freestyle = True
        freestyle_settings = self.scene.view_layers["View Layer"].freestyle_settings
        self._set_freestyle_settings(freestyle_settings)
        self.scene.use_nodes = True
        tree = self.scene.node_tree
        self._clean_node_tree_links()
        source_node = tree.nodes["Render Layers"]
        alphaover = self._get_or_add_alphaover_node()
        destination_node = tree.nodes["Composite"]
        tree.links.new(source_node.outputs["Freestyle"], alphaover.inputs[2])
        tree.links.new(alphaover.outputs["Image"], destination_node.inputs["Image"])

    def _set_freestyle_settings(self, freestyle_settings):
        freestyle_settings.as_render_pass = True
        lineset = freestyle_settings.linesets[0]
        lineset.linestyle.thickness = 1.65
        lineset.select_silhouette = True
        lineset.select_contour = True
        lineset.select_suggestive_contour = False  # False for ANNFASS
        lineset.select_ridge_valley = False  # False for ANNFASS
        lineset.select_crease = False  # can be False for ANNFASS
        lineset.select_border = False  # False for ANNFASS
        lineset.select_external_contour = False  # can be False for ANNFASS
        lineset.select_material_boundary = False
        lineset.select_edge_mark = False


class RidgeValleyRenderer(FreestyleRenderer):
    def __init__(self, scene, low_resolution=True, fast_and_cheap=True, transparent_background=False):
        super().__init__(scene, low_resolution, fast_and_cheap, transparent_background)
        self._set_freestyle()

    def _set_freestyle_settings(self, freestyle_settings):
        freestyle_settings.as_render_pass = True
        lineset = freestyle_settings.linesets[0]
        lineset.linestyle.thickness = 1.65
        lineset.select_silhouette = False
        lineset.select_contour = False
        lineset.select_suggestive_contour = False
        lineset.select_ridge_valley = True
        lineset.select_crease = False
        lineset.select_border = False
        lineset.select_external_contour = False
        lineset.select_material_boundary = False
        lineset.select_edge_mark = False


class CyclesRenderer(Renderer):
    def __init__(self, scene, low_resolution=True, fast_and_cheap=True, transparent_background=False):
        super().__init__(scene, low_resolution, fast_and_cheap, transparent_background)
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'GPU'
        self.scene.world.node_tree.nodes["Background"].inputs[0].default_value = DAYLIGHT_COLOR
        if fast_and_cheap:
            self.scene.cycles.max_bounces = 6
            self.scene.cycles.sample_clamp_indirect = 8


class EeveeRenderer(Renderer):
    def __init__(self, scene, low_resolution=True, fast_and_cheap=True, transparent_background=False):
        super().__init__(scene, low_resolution, fast_and_cheap, transparent_background)
        self.scene.render.engine = 'BLENDER_EEVEE'
        # self.scene.world.node_tree.nodes["Background"].inputs[0].default_value = DAYLIGHT_COLOR
        self.scene.world.node_tree.nodes["Background"].inputs[0].default_value = SUNLIGHT_COLOR


# class WorkbenchRenderer(Renderer):
#     def __init__(self, scene, low_resolution=True, fast_and_cheap=True):
#         super().__init__(scene, low_resolution, fast_and_cheap)
#         self.scene.render.engine = 'BLENDER_WORKBENCH'
#
#         screens = [bpy.context.screen]
#         for s in screens:
#             for spc in s.areas:
#                 if spc.type == "VIEW_3D":
#                     spc.spaces[0].shading.type = 'RENDERED'
#                     spc.spaces[0].shading.light = 'STUDIO'
#                     # spc.spaces[0].shading.light = 'MATCAP'
#                     # spc.spaces[0].shading.color_type = 'TEXTURE'
#                     spc.spaces[0].shading.color_type = 'MATERIAL'
#                     break  # we expect at most 1 VIEW_3D space
