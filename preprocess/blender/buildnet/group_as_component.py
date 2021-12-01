import argparse
import logging
import os.path
import sys
from os.path import dirname, join, realpath, exists, basename

STYLE_DIR = dirname(dirname(dirname(dirname(realpath(__file__)))))
sys.path.append(STYLE_DIR)
from common.utils import set_logger_file, parse_buildings_csv, STYLES, str2bool

from preprocess.blender.scene_utils import cleanup
from preprocess.blender.io_utils import load_obj, save_obj
from preprocess.blender.scene_utils import add_scene, add_world, get_scene, set_active_scene, remove_objects
from preprocess.blender.buildnet.group_helper import *

LOGGER = logging.getLogger(__name__)

STYLISTIC_ELEMENTS_THRESHOLD_JSON = join(STYLE_DIR, "resources", "BUILDNET_GROUP_THRESHOLDS_v2.json")
STYLISTIC_PARTS_THR_FACTORS = json.load(open(STYLISTIC_ELEMENTS_THRESHOLD_JSON))
OBB_CMD = join(STYLE_DIR, "preprocess/oriented_bounding_box/cgal_impl/cmake-build-release/OrientedBboxC")


# Scene 1 has the original (normalized) building composed by many components with textures and a camera to render
# Scene 2 has the (normalized) building loaded as ply format, i.e. one united mesh, to be used for ray tracing.
# Scene 3 has the (normalized) building loaded as obj format with objects having no materials, so that we use red.
add_scene()
add_world()
add_scene()
add_world()
scenes_str = ['Scene', 'Scene.001', 'Scene.002']
worlds_str = ['World', 'World.001', 'World.002']

obj_scene = get_scene(scenes_str[0])


def get_elements_with_style(scene):
    elements = []
    for obj in scene.objects:
        if "__" in obj.name:
            elements.append(obj.name)
    return elements


def main_group(obj_inp_file, groups_dir, debug, on_gpu, transparent_back):
    groups_file = join(groups_dir, "groups.json")
    group_colors_file = join(groups_dir, "g_colours.json")
    gr_prefix = "_grouped_" if not transparent_back else "tr_group_"
    if exists(groups_file) \
            and any(gr_prefix in f for f in os.listdir(groups_dir)) \
            and any(".ply" in f for f in os.listdir(groups_dir)):
        LOGGER.warning("Won't process {}".format(groups_dir))
        return
    log_file = join(groups_dir, "logs.log")

    # One scene with the loaded obj and the camera
    remove_objects(except_names=['Camera'])
    set_active_scene(obj_scene.name)
    load_obj(obj_inp_file)

    elements_with_style = get_elements_with_style(obj_scene)
    LOGGER.info("elements with style: {}".format(", ".join(elements_with_style)))
    if len(elements_with_style) == 0:
        LOGGER.warning("No elements with style ==> nothing to be done.")
        save_groups({}, groups_file)
        return

    if exists(groups_file):
        LOGGER.info("Groups exist and will be loaded for further processing.")
        all_groups = list(load_groups(groups_file).values())
        g_colors = load_colors(group_colors_file)
    else:
        all_groups, g_colors = create_groups(elements_with_style, group_colors_file, groups_file, debug)

    grouped_obj_file = join(groups_dir, "grouped.obj")
    if not any(gr_prefix in f for f in os.listdir(groups_dir)):
        LOGGER.info("Colorizing groups")
        colorize(all_groups, g_colors, remove_material=True, all_same=False)
        LOGGER.info("Rendering groups")
        visualize_groups(groups_dir, obj_scene, on_gpu=on_gpu, transparent_back=transparent_back)

    if not exists(grouped_obj_file):
        save_obj(grouped_obj_file)

    if not any(".ply" in f for f in os.listdir(groups_dir)):
        LOGGER.info("Creating ply files per group")
        group_ply_filenames = save_ply_groups(all_groups, groups_dir, obj_scene)
        for group_ply_filename in group_ply_filenames:
            print('{} "{}" >> "{}"'.format(OBB_CMD, group_ply_filename, log_file))
            os.system('{} "{}" >> "{}"'.format(OBB_CMD, group_ply_filename, log_file))  # "" is needed due to spaces
    cleanup(True)


def create_groups(elements_with_style, group_colors_file, groups_file, debug):
    LOGGER.info("Creating groups...")
    all_groups = []  # [ [group1_element1name, group1_element2name,...], [group2_element1name, ...], ...]
    for stylistic_part, threshold_factor in STYLISTIC_PARTS_THR_FACTORS.items():
        LOGGER.info(f"{stylistic_part}:{threshold_factor}")
        for style in STYLES:
            current_elements = [element for element in elements_with_style
                                if stylistic_part.lower() in element.lower()
                                and style.lower() in element.lower()]
            if len(current_elements) > 0:
                debug_dir = None
                if debug:
                    debug_dir = os.path.join(os.path.dirname(groups_file), "debug", stylistic_part, style)
                groups = ComponentsGrouper(LOGGER, factor=float(threshold_factor), debug_dir=debug_dir)\
                    .add_all(current_elements)\
                    .group()\
                    .get()
                LOGGER.info("{} groups of {}".format(len(groups), stylistic_part))
                all_groups += groups
    g_colors = get_colors(all_groups)
    save_colors(g_colors, group_colors_file)
    save_groups(all_groups, groups_file)
    return all_groups, g_colors


def process_building(input_file, output_dir, debug, on_gpu, transparent_back):
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info("grouping...")
    main_group(input_file, output_dir, debug, on_gpu, transparent_back)


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-root_data', type=str, default="/media/christina/Data/ANNFASS_data/BUILDNET_Buildings")
        parser.add_argument('-obj_dir', type=str, default="normalizedObj")
        parser.add_argument('-group_dir', type=str, default="groups")
        parser.add_argument('-buildings_csv', type=str, default="buildings.csv")
        parser.add_argument('-logs_dir', type=str, default="/media/christina/Data/ANNFASS_data/BUILDNET_Buildings/logs")
        parser.add_argument('-debug', type=str2bool, default=False)
        parser.add_argument('-on_gpu', type=str2bool, default=True)
        parser.add_argument('-transparent_back', type=str2bool, default=True)
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    if not exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = join(args.logs_dir, basename(args.buildings_csv).replace('.csv', '.log'))
    LOGGER = set_logger_file(_log_file, LOGGER)

    LOGGER.info("Starting...")

    LOGGER.info("root_data {}".format(args.root_data))
    LOGGER.info("obj_dir {}".format(args.obj_dir))
    LOGGER.info("group_dir {}".format(args.group_dir))
    LOGGER.info("buildings_csv {}".format(args.buildings_csv))

    buildings = parse_buildings_csv(join(args.root_data, args.buildings_csv))
    LOGGER.info("buildings: {}".format(buildings))

    for building in buildings:
        obj_file = join(args.root_data, args.obj_dir, building, "{}.obj".format(building))
        if not exists(obj_file):
            LOGGER.info("{} doesn't exist. Won't process.".format(obj_file))
            continue
        group_dir = join(args.root_data, args.group_dir, building)
        LOGGER.info("Processing {}".format(obj_file))
        process_building(obj_file, group_dir, args.debug, args.on_gpu, args.transparent_back)

    LOGGER.info("Ending...")


# /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender --background --python group_as_component.py -- -root_data /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings -obj_dir normalizedObj -group_dir group_debug -buildings_csv buildings_debug.csv -logs_dir /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/grouplogs -debug True -on_gpu False
