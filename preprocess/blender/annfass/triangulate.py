import argparse
import logging
import os
import sys

# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from preprocess.blender.scene_utils import cleanup
from preprocess.blender.mesh_utils import triangulate_all
from preprocess.blender.io_utils import load_fbx, save_obj_with_textures
from common.utils import set_logger_file, parse_buildings_with_style_csv


LOGGER = logging.getLogger(__file__)


def process_building(input_file, output_file):
    if os.path.exists(output_file):
        LOGGER.info("Output file exists. Won't process..")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleanup()
    LOGGER.info("loading...")
    load_fbx(input_file)
    LOGGER.info("triangulating...")
    triangulate_all()
    LOGGER.info("unit scaling...")
    LOGGER.info("saving...")
    save_obj_with_textures(output_file)


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-root_data', type=str, required=True)
        parser.add_argument('-raw_data', type=str, default="raw_data")
        parser.add_argument('-obj_out', type=str, default="normalizedObj")
        parser.add_argument('-buildings_csv', type=str, required=True)
        parser.add_argument('-logs_dir', type=str, default="/media/christina/Data/ANNFASS_data/ANNFASS_Buildings/logs")
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = os.path.join(args.logs_dir, os.path.basename(args.buildings_csv).replace('.csv', '.log'))
    LOGGER = set_logger_file(_log_file, LOGGER)

    LOGGER.info(f"Starting {os.path.realpath(__file__)}")

    LOGGER.info("root_data {}".format(args.root_data))
    LOGGER.info("raw_data {}".format(args.raw_data))
    LOGGER.info("obj_out {}".format(args.obj_out))
    LOGGER.info("buildings_csv {}".format(args.buildings_csv))

    buildings = parse_buildings_with_style_csv(args.buildings_csv)
    LOGGER.info("buildings: {}".format(buildings))

    for (style, building) in buildings:
        fbx_file = os.path.join(args.root_data, args.raw_data, style, building, "{}.fbx".format(building))
        if not os.path.exists(fbx_file):
            LOGGER.info("{} doesn't exist. Won't process.".format(fbx_file))
            continue
        obj_dir = os.path.join(args.root_data, args.obj_out, building)
        LOGGER.info("Processing {}".format(fbx_file))
        out_f = os.path.join(obj_dir, "{}.obj".format(building))
        process_building(fbx_file, out_f)

    LOGGER.info(f"End {os.path.realpath(__file__)}")


# blender --background --python triangulate.py -- -root_data /media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_semifinal -obj_out normalizedObjTmp -buildings_csv buildingstmp.csv -logs_dir /media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_semifinal/trianglogs.txt