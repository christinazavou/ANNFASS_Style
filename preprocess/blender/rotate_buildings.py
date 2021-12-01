import argparse
import logging
import os
import sys
from os.path import dirname, realpath, join

STYLE_DIR = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(STYLE_DIR)
from preprocess.blender.scene_utils import cleanup
from preprocess.blender.io_utils import load_obj, save_obj
from common.utils import parse_buildings_csv

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-root_data', type=str)
        parser.add_argument('-obj_dir_in', type=str, default="normalizedObj_refinedTextures")
        parser.add_argument('-obj_dir_out', type=str, default="normalizedObj_refinedTexturesRotated")
        parser.add_argument('-buildings_csv', type=str, default=None)
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    if args.buildings_csv == "" or args.buildings_csv is None:
        buildings = None
    else:
        buildings = parse_buildings_csv(join(args.root_data, args.buildings_csv))
    LOGGER.info("buildings: {}".format(buildings))

    for building in os.listdir(join(args.root_data, args.obj_dir_in)):
        if buildings is not None and building not in buildings:
            continue
        if os.path.isfile(join(args.root_data, args.obj_dir_in, building)) and building.endswith(".obj"):
            file_in = join(args.root_data, args.obj_dir_in, building)
            file_out = join(args.root_data, args.obj_dir_out, building)
        else:
            file_in = join(args.root_data, args.obj_dir_in, building, f"{building}.obj")
            file_out = join(args.root_data, args.obj_dir_out, building, f"{building}.obj")

        os.makedirs(os.path.dirname(file_out), exist_ok=True)

        cleanup(True)
        load_obj(file_in, rotated=True)
        save_obj(file_out, axis_up='Y')


# blender --background --python rotate_buildings.py -- -root_data /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings -obj_dir_in normalizedObj_refinedTextures -out_dir normalizedObj_refinedTexturesRotated --buildings_csv ...

# blender --background --python rotate_buildings.py -- -root_data /media/graphicslab/BigData/zavou/ANNFASS_CODE/proj_style_data/data/building_yu_fullymine -obj_dir_in models -out_dir models_rotated
