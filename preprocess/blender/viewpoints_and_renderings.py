import argparse
import json
import logging
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector

OBB_CMD = "../oriented_bounding_box/cgal_impl/cmake-build-release/OrientedBboxC"

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(SOURCE_DIR)
from common.utils import set_logger_file, parse_buildings_csv, BUILDNET_STYLISTIC_ELEMENTS, ANNFASS_STYLISTIC_ELEMENTS, str2bool
from preprocess.blender.mesh_utils import overall_scene_mesh, obj_mesh
from preprocess.blender.raycast_utils import RayCastChecker
from preprocess.blender.bounding_utils import sphere_bound
from preprocess.blender.scene_utils import remove_obj, set_active_scene, add_scene, add_world, cleanup
from preprocess.blender.camera_utils import camera_distance_from_sphere, rotate_towards_target, get_or_add_camera
from preprocess.blender.io_utils import load_ply, load_obj
from preprocess.blender.space_utils import vector_is_facing_upwards_or_downwards
from preprocess.blender.renderer_utils import EeveeRenderer, FreestyleRenderer, Renderer, CyclesRenderer
from preprocess.blender.colour_utils import RED_COLOR


class GroupMultiRendererGivenObb(object):
    def __init__(self, render_dir, view_file, groups_dir, ply_obb_file,
                 components, group_name,
                 scene_multiple, scene_union):
        self.render_dir = os.path.join(render_dir, group_name)
        self.view_file = view_file
        self.groups_dir = groups_dir
        self.objs = components
        self.group_name = group_name
        self.distance_factors = [0.5, 1, 1.5]
        self.scene_union = scene_union
        self.scene_multiple = scene_multiple
        self.bound_box = None
        self.box_bmesh = None
        self.obj = None
        self.view_points = []
        self.camera_distance = None
        self.overall_mesh = None
        self.rtc = None
        self.obb_file = ply_obb_file
        self.horizontal_directions = []
        horizontal_vector = Vector((0, 1, 0))
        self.horizontal_directions.append(horizontal_vector)
        for i in range(11):
            self.horizontal_directions.append(horizontal_vector @ Matrix.Rotation(math.radians(30+30*i), 4, (0, 0, 1)))
        self.vertical_directions = [Vector((0, 0, 1)), Vector((0, 0, -1))]
    #
    def _set_overall_mesh(self):
        self.overall_mesh = overall_scene_mesh(self.scene_union)
        self.rtc = RayCastChecker(self.overall_mesh)
    #
    def _unset_overall_mesh(self):
        self.overall_mesh = None
        self.rtc = None
    #
    def _set_camera_distance(self):
        # doesn't matter which scene is active as long as we pass the object references to be bounded and we remove the
        # added sphere
        bound_sphere, radius = sphere_bound(self.objs)
        remove_obj(bound_sphere)
        self.camera_distance = camera_distance_from_sphere(radius, bpy.data.cameras['Camera'])
        LOGGER.debug("camera distance: {}".format(self.camera_distance))
    #
    def _set_object(self, obj):
        # obj is a whole group
        LOGGER.debug("using obj {} for naming files".format(obj.name))
        self.obj = obj
        group_obb_f = os.path.join(self.groups_dir, "{}_obb.ply".format(self.group_name))
        if not os.path.exists(group_obb_f):
            LOGGER.info("Will not render {} as {} doesn't exist".format(self.group_name, group_obb_f))
            self.bound_box = None
            return
        set_active_scene(self.scene_union.name)
        load_ply(group_obb_f)
        self.bound_box = self.scene_union.objects["{}_obb".format(self.group_name)]
        self.box_bmesh = obj_mesh(self.bound_box)
        # the box is in scene_union so we can hit it on ray cast
    #
    def _unset_object(self):
        if self.bound_box is not None:
            remove_obj(self.bound_box)
        self.box_bmesh = None
        self.bound_box = None
        self.obj = None
    #
    def try_with_faces(self):
        for face_idx, face in enumerate(self.box_bmesh.faces):
            face_center = self.bound_box.matrix_world @ face.calc_center_median()
            face_norm = self.bound_box.matrix_world @ face.normal
            if face_norm == Vector((0, 0, 0)):
                LOGGER.debug("face norm is zero at face_center {}. Probably the box is flat.".format(face_center))
            else:
                LOGGER.debug("face_center: {}, face_norm: {}".format(face_center, face_norm))
                if not vector_is_facing_upwards_or_downwards(face_norm, logger=LOGGER):  # we have a face to check
                    _ = self._add_view_points(face_center, face_idx, face_norm, "0x0y")
                    # check from slightly left/right side
                    for x_angle in [-20, 20, 40, -40]:
                        rotated_direction = face_norm @ Matrix.Rotation(math.radians(x_angle), 4, (0, 0, 1))
                        new_count = self._add_view_points(face_center, face_idx, rotated_direction, "{}x{}y".format(x_angle, 0))
                        if new_count == 0:
                            for y_angle in [-30, 30]:
                                rotated_direction = face_norm @ Matrix.Rotation(math.radians(x_angle), 4, (0, 0, 1)) @ Matrix.Rotation(math.radians(y_angle), 4, (1, 0, 0))
                                _ = self._add_view_points(face_center, face_idx, rotated_direction, "{}x{}y".format(x_angle, y_angle))
    #
    def _calculate_view_points(self):
        self._set_camera_distance()
        for obj in self.objs[0:1]: #assuming grouped object so will just use first object of group TODO:is it ok?
            self._set_object(obj)
            if self.bound_box is None:
                self._unset_object()
                continue
            self._set_overall_mesh()
            self.try_with_faces()
            self._unset_overall_mesh()
        self.camera_distance = None
    #
    def _add_view_points(self, look_at_point, look_at_idx, direction, angle):
        found_viewpoints = 0
        for dist_idx, dist_factor in enumerate(self.distance_factors):
            LOGGER.debug("dir: {}, dist_f: {}, camera_d: {}".format(-direction, dist_factor, self.camera_distance))
            location = look_at_point + dist_factor * direction * self.camera_distance
            LOGGER.debug("location: {}".format(location))
            if self.rtc.hits_mesh_in_all_directions(location, self.horizontal_directions):
                LOGGER.debug("location {} is inside building".format(location))
                continue
            # there is a direction that doesn't hit building .. but still could be a spot
            # with multiple windows .. thus we can check also if there is floor and ceiling there
            if self.rtc.hits_mesh_in_all_directions(location, self.vertical_directions):
                LOGGER.debug("location {} is inside building".format(location))
                continue
            if self.rtc.shot_vertex(location, -direction, look_at_point):
                self.view_points.append((
                    self.obj.name, look_at_idx, dist_idx,
                    (look_at_point.x, look_at_point.y, look_at_point.z),
                    (direction.x, direction.y, direction.z),
                    (location.x, location.y, location.z),
                    angle
                ))
                found_viewpoints += 1
            else:
                break  # if small distance doesn't hit it then bigger distance won't hit it either
        return found_viewpoints
    #
    def run(self, camera, bl_renderer):
        if not os.path.exists(self.view_file):
            LOGGER.info("Will calculate view points")
            self._calculate_view_points()
            self._save()
        else:
            LOGGER.info("Will load view points")
            self._load()
        set_active_scene(self.scene_multiple.name)
        self._render(camera, bl_renderer)
    #
    def _render(self, camera, bl_renderer):
        LOGGER.info("Will render from {} viewpoints".format(len(self.view_points)))
        s_time = time.time()
        for obj_name, face_idx, dist_idx, face_center, direction, camera_location, angle in self.view_points:
            camera.location = camera_location
            self._render_view_point(camera, Vector(face_center), obj_name, face_idx, dist_idx, angle, bl_renderer)
        e_time = time.time()
        LOGGER.debug("took {} seconds to render the group".format(round(e_time-s_time, 5)))
    #
    def _render_view_point(self, camera, look_at_coord, objname, face_idx, dist_idx, angle, bl_renderer):
        rotate_towards_target(camera, look_at_coord)
        fn = os.path.join(self.render_dir, "{}_{}_{}_{}.jpg".format(objname, face_idx, dist_idx, angle))
        bl_renderer(fn)
    #
    def _save(self):
        os.makedirs(os.path.dirname(self.view_file), exist_ok=True)
        with open(self.view_file, "w") as fout:
            json.dump(self.view_points, fout)  # TODO save camera rotation ?
    #
    def _load(self):
        with open(self.view_file, "r") as fin:
            self.view_points = json.load(fin)
    #
    @staticmethod
    def render_given_viewpoints(camera, bl_renderer, viewpoints, render_dir, suffix=""):
        LOGGER.info("Will render from {} viewpoints".format(len(viewpoints)))
        s_time = time.time()
        for obj_name, face_idx, dist_idx, face_center, direction, camera_location, angle in viewpoints:
            camera.location = camera_location
            rotate_towards_target(camera, Vector(face_center))
            fn = os.path.join(render_dir, "{}_{}_{}_{}_{}.jpg".format(obj_name, face_idx, dist_idx, angle, suffix))
            bl_renderer(fn)
        e_time = time.time()
        LOGGER.debug("took {} seconds to render the group".format(round(e_time-s_time, 5)))


def get_group_objects_in_scene(group, scene):
    the_group = []
    for name in group:
        if name in scene.objects:
            the_group.append(scene.objects[name])
        else:
            if name.split(".")[0] in scene.objects:
                the_group.append(scene.objects[name.split(".")[0]])
            else:
                for obj in scene.objects:
                    if name.split(".")[0] in obj.name:
                        the_group.append(obj)
    return the_group


def colour_group_red(scene, group_objects, red_mat):
    for obj in scene.objects:
        if obj.type == 'MESH':
            if obj in group_objects:
                obj.active_material = red_mat
            else:
                obj.active_material = None


def render_groups(groups, duplicates, output_dir, views_dir, scene_multiple,
                  backface_culling=True):
    LOGGER.debug("render_groups.groups : {}".format(groups))
    if backface_culling:
        Renderer.set_backface_culling(scene_multiple)
    #
    camera = get_or_add_camera()
    if RENDER_MODE == 1:
        bl_renderer = FreestyleRenderer(scene_multiple)
    else:
        if USE_GPU_CYCLES:
            bl_renderer = CyclesRenderer(scene_multiple)
        else:
            bl_renderer = EeveeRenderer(scene_multiple)
    #
    for unique_component, duplicate_components in duplicates.items():
        group_component_id = unique_component.replace("style_mesh_group", "")
        group_component_id = group_component_id.split("_")[0]
        group = groups[group_component_id]
        i = int(group_component_id)
        LOGGER.info("processing group {}".format(i))
        the_group = get_group_objects_in_scene(group, scene_multiple)
        LOGGER.debug("using GroupMultiRendererGivenObb & group {}".format(i))
        gmr = GroupMultiRendererGivenObb(output_dir, os.path.join(views_dir, "group_{}_views_selected.json".format(i)),
                                         None, None, the_group, "group_{}".format(i), scene_multiple, scene_multiple)
        gmr.run(camera, bl_renderer)


def viewpoints_groups(groups, duplicates, views_dir, groups_dir, ply_obb_file, scene_union_str,
                      scene_multiple_copy_str, backface_culling=True):
    LOGGER.debug("viewpoints_groups.groups : {}".format(groups))
    if backface_culling:
        Renderer.set_backface_culling(bpy.data.scenes[scene_multiple_copy_str])
    #
    camera = get_or_add_camera()
    red_mat = bpy.data.materials.new(name="RedColor")
    red_mat.diffuse_color = RED_COLOR
    # red_camera = bpy.data.objects.new("RedCamera", bpy.data.cameras.new("Camera"))
    # scene_multiple_copy.collection.objects.link(red_camera)
    bpy.data.scenes[scene_multiple_copy_str].camera = camera  # set active camera in scene FIXME should be red_camera?
    # red_camera.location = camera.location
    # red_camera.rotation_euler = camera.rotation_euler
    if USE_GPU_CYCLES:
        bl_renderer_red = CyclesRenderer(bpy.data.scenes[scene_multiple_copy_str])
    else:
        bl_renderer_red = EeveeRenderer(bpy.data.scenes[scene_multiple_copy_str])
    #
    for unique_component, duplicate_components in duplicates.items():
        group_component_id = unique_component.replace("style_mesh_group", "")
        group_component_id = group_component_id.split("_")[0]
        group = groups[group_component_id]
        i = int(group_component_id)
        LOGGER.info("processing group {}".format(i))
        the_group_red = get_group_objects_in_scene(group, bpy.data.scenes[scene_multiple_copy_str])
        colour_group_red(bpy.data.scenes[scene_multiple_copy_str], the_group_red, red_mat)
        LOGGER.debug("using GroupMultiRendererGivenObb & group {}".format(i))
        gmr = GroupMultiRendererGivenObb(views_dir, os.path.join(views_dir, "group_{}_views.json".format(i)),
                                         groups_dir, ply_obb_file, the_group_red, "group_{}".format(i),
                                         bpy.data.scenes[scene_multiple_copy_str], bpy.data.scenes[scene_union_str])
        gmr.run(camera, bl_renderer_red)


def obb_building(ply_filename):
    obb_file = ply_filename.replace(".ply", "_obb.ply")
    if not os.path.exists(obb_file):
        print('{} "{}"'.format(OBB_CMD, ply_filename))
        os.system('{} "{}"'.format(OBB_CMD, ply_filename))
    return obb_file


def main_render(obj_scene_str, obj_inp_file, output_dir, groups_dir, duplicates_file, views_dir, backface_culling=True):
    if os.path.exists(output_dir):
        LOGGER.warning("{} exists..won't process".format(output_dir))
        return  # don't override
    groups_file = os.path.join(groups_dir, "groups.json")
    if not os.path.exists(groups_file):
        LOGGER.warning("{} doesn't exist.. so won't proceed to in rendering".format(groups_file))
        return
    cleanup(materials=True, except_names=['Camera'])
    set_active_scene(obj_scene_str)
    load_obj(obj_inp_file)
    with open(groups_file, "r") as fin:
        groups = json.load(fin)
    with open(duplicates_file, "r") as fin:
        duplicates = json.load(fin)
    render_groups(groups, duplicates, output_dir, views_dir, bpy.data.scenes[obj_scene_str], backface_culling)
    cleanup(materials=True, except_names=['Camera'])


def main_view_points(ply_scene_str, obj_copy_scene_str,
                     obj_inp_file, ply_inp_file,
                     viewpoints_dir, groups_dir, duplicates_file, backface_culling=True):
    if os.path.exists(viewpoints_dir):
        LOGGER.warning("{} exists..won't process".format(viewpoints_dir))
        return  # don't override
    groups_file = os.path.join(groups_dir, "groups.json")
    if not os.path.exists(groups_file):
        LOGGER.warning("{} doesn't exist.. so won't proceed to in rendering".format(groups_file))
        return
    if not os.path.exists(duplicates_file):
        LOGGER.warning("{} doesn't exist.. so won't proceed to in rendering".format(duplicates_file))
        return
    obb_file = obb_building(ply_inp_file)
    cleanup(materials=True, except_names=['Camera'])
    with open(groups_file, "r") as fin:
        groups = json.load(fin)
    with open(duplicates_file, "r") as fin:
        duplicates = json.load(fin)
    set_active_scene(ply_scene_str)
    load_ply(ply_inp_file)
    set_active_scene(obj_copy_scene_str)
    load_obj(obj_inp_file)
    viewpoints_groups(groups, duplicates, viewpoints_dir, groups_dir, obb_file, ply_scene_str, obj_copy_scene_str, backface_culling)
    cleanup(materials=True, except_names=['Camera'])


def make_scenes():
    # Scene 1 has the original (normalized) building composed by many components with textures and a camera to render
    # Scene 2 has the (normalized) building loaded as ply format, i.e. one united mesh, to be used for ray tracing.
    # Scene 3 has the (normalized) building loaded as obj format with objects having no materials, so that we use red.
    add_scene()
    add_world()
    add_scene()
    add_world()
    _scenes_str = ['Scene', 'Scene.001', 'Scene.002']
    _worlds_str = ['World', 'World.001', 'World.002']
    # link worlds to scenes
    bpy.data.scenes[1].world = bpy.data.worlds[1]
    bpy.data.scenes[2].world = bpy.data.worlds[2]
    return _scenes_str, _worlds_str


def run_building(building, obj_file, ply_file, renderings_dir, views_dir, groups_dir, duplicates_file):
    if not os.path.exists(obj_file):
        LOGGER.warning("Obj file for building {} doesn't exist. Nothing to be done.".format(building))
        return
    if not os.path.exists(ply_file):
        LOGGER.warning("Ply file for building {} doesn't exist. Nothing to be done.".format(building))
        return
    if not os.path.exists(duplicates_file):
        LOGGER.warning("Duplicates file for building {} doesn't exist. Nothing to be done.".format(building))
        return
    LOGGER.info("Processing {}".format(building))
    if MODE == 1:
        main_view_points(scenes_strs[1], scenes_strs[2],
                         obj_file, ply_file, views_dir, groups_dir, duplicates_file,
                         backface_culling=True)
    else:
        main_render(scenes_strs[0], obj_file, renderings_dir, groups_dir, duplicates_file, views_dir,
                    backface_culling=True)


def loop_over(renders_dirs, groups_dirs, unique_dir, viewpoints_dir, objs_dirs, plys_dirs, buildings_csv):
    if MODE == 1:
        os.makedirs(viewpoints_dir, exist_ok=True)
    if MODE == 2:
        os.makedirs(renders_dirs, exist_ok=True)
    buildings = parse_buildings_csv(buildings_csv)
    LOGGER.info("buildings = {}".format(buildings))
    for building in buildings:
        obj_file = os.path.join(objs_dirs, building, "{}.obj".format(building))
        ply_file = os.path.join(plys_dirs, building, "{}.ply".format(building))
        duplicates_file = os.path.join(unique_dir, building, "duplicates.json")
        renderings_dir = os.path.join(renders_dirs, RENDER_SUB_DIR, building)
        views_dir = os.path.join(viewpoints_dir, building)
        groups_dir = os.path.join(groups_dirs, building)
        run_building(building, obj_file, ply_file, renderings_dir, views_dir, groups_dir, duplicates_file)


if __name__ == '__main__':

    LOGGER = logging.getLogger(__name__)

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        print("argv to parse: {}".format(argv))
        parser = argparse.ArgumentParser()
        parser.add_argument('-MODE', type=int, default=1, help="1: create viewpoints, 2: render")
        parser.add_argument('-RENDER_MODE', type=int, default=1, help="1: freestyle, 2: materials")
        parser.add_argument('-GROUPS_DIR', type=str, default="groups")
        parser.add_argument('-RENDERS_DIR', type=str, default="renderings")
        parser.add_argument('-VIEWPOINTS_DIR', type=str, default="viewpoints")
        parser.add_argument('-UNIQUE_DIR', type=str, default="unique_point_clouds")
        parser.add_argument('-OBJ_DIR', type=str, default="normalizedObj")
        parser.add_argument('-PLY_DIR', type=str, default="normalizedPly")
        parser.add_argument('-BUILDINGS_CSV', type=str)
        parser.add_argument('-LOGS_DIR', type=str)
        parser.add_argument('-USE_GPU_CYCLES', type=str2bool, default=False)
        args = parser.parse_known_args(argv)[0]

    MODE = args.MODE
    RENDER_MODE = args.RENDER_MODE
    RENDER_SUB_DIR = "freestyle" if RENDER_MODE == 1 else "materials_on_daylight"
    USE_GPU_CYCLES = args.USE_GPU_CYCLES  # note that with cycles engine we get some black triangles artifacts ...
    # thus use it only for viewpoints

    os.makedirs(args.LOGS_DIR, exist_ok=True)
    log_file = os.path.join(args.LOGS_DIR, f'{os.path.basename(os.path.realpath(__file__))}.log')
    print("logs in ", log_file)
    set_logger_file(log_file, LOGGER)

    LOGGER.info("Starting...")

    LOGGER.info("MODE = {}, RENDER_MODE = {}, USE_GPU_CYCLES = {}".format(MODE, RENDER_MODE, USE_GPU_CYCLES))
    LOGGER.info("GROUPS_DIR = {}".format(args.GROUPS_DIR))
    LOGGER.info("RENDERS_DIR = {}".format(args.RENDERS_DIR))
    LOGGER.info("VIEWPOINTS_DIR = {}".format(args.VIEWPOINTS_DIR))
    LOGGER.info("UNIQUE_DIR = {}".format(args.UNIQUE_DIR))
    LOGGER.info("OBJ_DIR = {}".format(args.OBJ_DIR))
    LOGGER.info("PLY_DIR = {}".format(args.PLY_DIR))
    LOGGER.info("BUILDINGS_CSV = {}".format(args.BUILDINGS_CSV))
    LOGGER.info("LOGS_DIR = {}".format(args.LOGS_DIR))

    is_buildnet = True if "buildnet" in args.GROUPS_DIR.lower() else False
    if is_buildnet:
        STYLISTIC_ELEMENTS = BUILDNET_STYLISTIC_ELEMENTS
    else:
        STYLISTIC_ELEMENTS = ANNFASS_STYLISTIC_ELEMENTS

    scenes_strs, worlds_strs = make_scenes()

    loop_over(args.RENDERS_DIR, args.GROUPS_DIR, args.UNIQUE_DIR, args.VIEWPOINTS_DIR, args.OBJ_DIR, args.PLY_DIR,
              args.BUILDINGS_CSV)

    LOGGER.info("Ending...")


# /home/graphicslab/Downloads/software/blender-2.93.5-linux-x64/blender --background --python viewpoints_and_renderings.py -- -GROUPS_DIR /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17 -RENDERS_DIR /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_renderings -VIEWPOINTS_DIR /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_viewpoints -BUILDINGS_CSV /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings_torender.csv -LOGS_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/logs_june -UNIQUE_DIR /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds -OBJ_DIR /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj -PLY_DIR /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedPly

# /home/maverkiou/blender-2.91.0-linux64/blender -b -noaudio --python viewpoints_and_renderings.py -- -GROUPS_DIR /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/groups_june17 -RENDERS_DIR /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/groups_june17_renderings -VIEWPOINTS_DIR /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/groups_june17_viewpoints -BUILDINGS_CSV /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/buildings_torender.csv -LOGS_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/logs_june -UNIQUE_DIR /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/groups_june17_unique_point_clouds -OBJ_DIR /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/normalizedObj -PLY_DIR /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/normalizedPly -RENDER_MODE 2 -MODE 2

# note: using /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender or /home/graphicslab/Downloads/software/blender-2.82-linux64/blender fails
