import argparse
import json
import logging
import os
import sys
from os.path import dirname, realpath, join, basename, exists

SOURCE_DIR = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(SOURCE_DIR)
from common.utils import parse_buildings_csv, set_logger_file, str2bool
from preprocess.blender.renderer_utils import EeveeRenderer
from preprocess.blender.scene_utils import cleanup, bpy
from preprocess.blender.io_utils import load_obj


LOGGER = logging.getLogger(__name__)


def run_component(file_in):
    rendering_files = os.listdir(dirname(file_in))
    if args.TRANSPARENT_BACK:
        rendering_files = [f for f in rendering_files if f.endswith(".png") and f.startswith("tr_img")]
    else:
        rendering_files = [f for f in rendering_files if f.endswith(".png") and f.startswith("img")]
    if len(rendering_files) == 6:
        LOGGER.info(f"renderings done already for {file_in}")
        return  # renderings done already
    cleanup(materials=True, except_names=['Camera', 'Light'])
    load_obj(file_in)
    objects = []
    for obj in bpy.data.objects:
        if obj.name != 'Light' and obj.name != 'Camera':
            obj.hide_render = False
            objects.append(obj)
        else:
            obj.hide_render = True
    renderer.multi_view(objects, dirname(file_in), "tr_img", bpy.data.objects['Camera'])


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        print("argv to parse: {}".format(argv))
        parser = argparse.ArgumentParser()
        parser.add_argument('-RENDER_MODE', type=int, default=1, help="1: freestyle, 2: materials")
        parser.add_argument('-ROOT_DIR', type=str)
        parser.add_argument('-COMPONENTS_DIR', type=str, default="unified_normalized_components")
        parser.add_argument('-UNIQUE_DIR', type=str, default="unique_point_clouds")
        parser.add_argument('-BUILDINGS_CSV', type=str)
        parser.add_argument('-TRANSPARENT_BACK', type=str2bool, default=True)
        parser.add_argument('-LOGS_DIR', type=str)
        args = parser.parse_known_args(argv)[0]

    if not os.path.exists(args.LOGS_DIR):
        os.makedirs(args.LOGS_DIR, exist_ok=True)
    log_file = os.path.join(args.LOGS_DIR, f'{basename(__file__)}.log')
    print("logs in ", log_file)
    set_logger_file(log_file, LOGGER)

    renderer = EeveeRenderer(bpy.data.scenes[0], transparent_background=args.TRANSPARENT_BACK)

    buildings = parse_buildings_csv(args.BUILDINGS_CSV)

    for building in buildings:
        idx = 0
        unique_file = join(args.ROOT_DIR, args.UNIQUE_DIR, building, "duplicates.json")
        if not exists(unique_file):
            LOGGER.info(f"{unique_file} doesnt exist...")
            continue
        with open(unique_file, "r") as fin:
            unique_components = json.load(fin).keys()
            LOGGER.info(unique_components)
        for component in unique_components:
            obj_file = join(args.ROOT_DIR, args.COMPONENTS_DIR, building, component, "model.obj")
            if exists(obj_file):
                idx += 1
                run_component(obj_file)
            else:
                LOGGER.info(f"{obj_file} doesn't exist...")
        LOGGER.info(f"processed {idx} components for {building}")


# blender --background --python render_components.py -- -ROOT_DIR /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings -COMPONENTS_DIR unified_normalized_components -UNIQUE_DIR newgroupv2_unique_point_clouds -BUILDINGS_CSV /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings_debug.csv -LOGS_DIR /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/mylogs
