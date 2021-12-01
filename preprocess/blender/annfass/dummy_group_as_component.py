import argparse
import logging
import sys
from os.path import dirname, realpath, join

import bpy

STYLE_DIR = dirname(dirname(dirname(dirname(realpath(__file__)))))
sys.path.append(STYLE_DIR)
from common.utils import set_logger_file, parse_buildings_csv

from preprocess.blender.scene_utils import cleanup
from preprocess.blender.io_utils import load_obj
from preprocess.blender.buildnet.group_helper import *


OBB_CMD = join(STYLE_DIR, "preprocess/oriented_bounding_box/cgal_impl/cmake-build-release/OrientedBboxC")

LOGGER = logging.getLogger(__file__)

obj_scene = bpy.context.scene


def get_elements_with_style(scene):
    elements = []
    for obj in scene.objects:
        if "__" in obj.name:
            elements.append(obj.name)
    return elements


def main_group(groups_dir):
    groups_file = os.path.join(groups_dir, "groups.json")
    log_file = join(groups_dir, "logs.log")

    if os.path.exists(groups_file):
        LOGGER.info("Output file exists. Won't override..")
        all_groups = list(load_groups(groups_file).values())
    else:
        elements_with_style = get_elements_with_style(obj_scene)
        LOGGER.info("elements with style: {}".format(", ".join(elements_with_style)))
        if len(elements_with_style) == 0:
            LOGGER.warning("No elements with style ==> nothing to be done.")
            save_groups({}, groups_file)
            return
        if not os.path.exists(groups_file):
            all_groups = create_groups(elements_with_style, groups_file)

    if not any(".ply" in f for f in os.listdir(groups_dir)):
        LOGGER.info("Creating ply files per group")
        group_ply_filenames = save_ply_groups(all_groups, groups_dir, obj_scene)
        for group_ply_filename in group_ply_filenames:
            print('{} "{}" >> "{}"'.format(OBB_CMD, group_ply_filename, log_file))
            os.system('{} "{}" >> "{}"'.format(OBB_CMD, group_ply_filename, log_file))  # "" is needed due to spaces


def create_groups(elements_with_style, groups_file):
    LOGGER.info("Creating dummy groups...")
    all_groups = []
    for i, element in enumerate(elements_with_style):
        all_groups.append([element])
    save_groups(all_groups, groups_file)
    return all_groups


def process_building(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cleanup(materials=True)
    LOGGER.info("loading...")
    load_obj(input_file)
    LOGGER.info("grouping...")
    main_group(output_dir)


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-obj_dir', type=str, required=True)
        parser.add_argument('-group_dir', type=str, required=True)
        parser.add_argument('-buildings_csv', type=str, required=True)
        parser.add_argument('-logs_dir', type=str, default="groups_logs")
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = os.path.join(args.logs_dir, os.path.basename(args.buildings_csv).replace('.csv', '.log'))
    LOGGER = set_logger_file(_log_file, LOGGER)

    LOGGER.info(f"Starting {os.path.realpath(__file__)}")

    LOGGER.info("obj_dir {}".format(args.obj_dir))
    LOGGER.info("group_dir {}".format(args.group_dir))
    LOGGER.info("buildings_csv {}".format(args.buildings_csv))

    buildings = parse_buildings_csv(args.buildings_csv)
    LOGGER.info("buildings: {}".format(buildings))

    for building in buildings:
        obj_file = os.path.join(args.obj_dir, building, "{}.obj".format(building))
        if not os.path.exists(obj_file):
            LOGGER.info("{} doesn't exist. Won't process.".format(obj_file))
            continue
        group_dir = os.path.join(args.group_dir, building)
        LOGGER.info("Processing {}".format(obj_file))
        process_building(obj_file, group_dir)

    LOGGER.info(f"End {os.path.realpath(__file__)}")


# blender --background --python dummy_group_as_component.py -- -obj_out /media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/normalizedObj -buildings_csv /media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/buildings.csv -logs_dir /media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/groups_logs
