import argparse
import json
import logging
import os
import sys
from os.path import dirname, realpath, exists, join, basename

STYLE_DIR = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(STYLE_DIR)
from common.utils import set_logger_file, parse_buildings_csv, str2bool
from preprocess.blender.scene_utils import select_objects, cleanup, bpy
from preprocess.blender.io_utils import load_obj, save_obj
from preprocess.blender.mesh_utils import normalize_obj


LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-root_data', type=str)
        parser.add_argument('-obj_dir', type=str, default="normalizedObj")
        parser.add_argument('-group_dir', type=str, default="groups")
        parser.add_argument('-out_dir', type=str, default="unified_normalized_components")
        parser.add_argument('-buildings_csv', type=str, default="buildings.csv")
        parser.add_argument('-logs_dir', type=str)
        parser.add_argument('-unify', type=str2bool, default=True)
        parser.add_argument('-normalize', type=str2bool, default=True)
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    if not exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = join(args.logs_dir, f'{basename(__file__)}.log')
    LOGGER = set_logger_file(_log_file, LOGGER)

    LOGGER.info("Starting...")

    LOGGER.info("root_data {}".format(args.root_data))
    LOGGER.info("obj_dir {}".format(args.obj_dir))
    LOGGER.info("group_dir {}".format(args.group_dir))
    LOGGER.info("buildings_csv {}".format(args.buildings_csv))

    buildings = parse_buildings_csv(join(args.root_data, args.buildings_csv))
    LOGGER.info("buildings: {}".format(buildings))

    for idx, building in enumerate(buildings):
        cleanup(True)
        print(f"processing building {building}")
        obj_file_in = join(args.root_data, args.obj_dir, building, f"{building}.obj")
        groups_file_in = join(args.root_data, args.group_dir, building, "groups.json")
        if not exists(obj_file_in) or not exists(groups_file_in):
            continue
        loaded_obj = False
        with open(groups_file_in, "r") as fin:
            groups = json.load(fin)
        for group_id, group_components in groups.items():
            print(f"group: {group_id}: {group_components}")
            unique_g_component = f"style_mesh_group{group_id}_{group_components[0]}"
            obj_file_out = join(args.root_data, args.out_dir, building, unique_g_component, "model.obj")
            if not exists(obj_file_out):
                if not loaded_obj:
                    load_obj(obj_file_in)
                loaded_obj = True
                os.makedirs(dirname(obj_file_out), exist_ok=True)
                select_objects(bpy.context.scene, group_components)
                if args.unify:
                    # in order to be able to join .. one object of the ones to be joined must be active
                    # (i.e. yellow highlight in blender; not orange); note that only one object is active at any time
                    # set the first object as active so than new mesh has the name of first group object
                    bpy.context.view_layer.objects.active = bpy.data.objects[group_components[0]]
                    bpy.ops.object.join()
                    #now that it's one object I can bring it to center & scale it
                    if args.normalize:
                        obj = bpy.context.active_object
                        normalize_obj(obj)
                save_obj(obj_file_out, use_selection=True, axis_up='Y')


# blender --background --python unify_and_normalize_components.py -- -root_data /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings -obj_dir normalizedObj -out_dir unified_normalized_components -buildings_csv buildings_debug.csv -group_dir group_debugeevee -unify True -normalize True -logs_dir /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/logsunified
