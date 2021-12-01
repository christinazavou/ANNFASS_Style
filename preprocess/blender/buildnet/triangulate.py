import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from preprocess.blender.scene_utils import cleanup
from preprocess.blender.mesh_utils import triangulate_all, clean_mesh
from preprocess.blender.io_utils import load_dae, save_obj_with_textures
from common.utils import set_logger_file, parse_buildings_csv


LOGGER = logging.getLogger(name="triangulate")


def process_building(input_file, output_file):
    if os.path.exists(output_file):
        LOGGER.info("Output file exists. Won't process..")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleanup()
    LOGGER.info("loading...")
    load_dae(input_file)
    print("removing loose...")
    clean_mesh()
    LOGGER.info("triangulating...")
    triangulate_all()
    LOGGER.info("saving...")
    save_obj_with_textures(output_file)


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-root_data', type=str, default="/media/christina/Data/ANNFASS_data/BUILDNET_Buildings")
        parser.add_argument('-raw_data', type=str, default="raw_data/colladaFiles")
        parser.add_argument('-obj_out', type=str, default="triangledObj")
        parser.add_argument('-buildings_csv', type=str, default="buildings.csv")
        parser.add_argument('-logs_dir', type=str, default="/media/christina/Data/ANNFASS_data/BUILDNET_Buildings/logs")
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = os.path.join(args.logs_dir, os.path.basename(args.buildings_csv).replace('.csv', '.log'))
    LOGGER = set_logger_file(_log_file, LOGGER)

    LOGGER.info("Starting...")

    LOGGER.info("root_data {}".format(args.root_data))
    LOGGER.info("raw_data {}".format(args.raw_data))
    LOGGER.info("obj_out {}".format(args.obj_out))
    LOGGER.info("buildings_csv {}".format(args.buildings_csv))

    buildings = parse_buildings_csv(os.path.join(args.root_data, args.buildings_csv))
    LOGGER.info("buildings: {}".format(buildings))

    for building in buildings:
        dae_file = os.path.join(args.root_data, args.raw_data, f"{building}.dae")
        if not os.path.exists(dae_file):
            LOGGER.info("{} doesn't exist. Won't process.".format(dae_file))
            continue
        obj_dir = os.path.join(args.root_data, args.obj_out, building)
        LOGGER.info("Processing {}".format(dae_file))
        out_f = os.path.join(obj_dir, "{}.obj".format(building))
        process_building(dae_file, out_f)

    LOGGER.info("Ending...")


# /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender --background --python triangulate.py -- -root_data /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings -obj_out triangledObjStylisticTemples -buildings_csv buildings_temples_with_style.csv -raw_data raw_data/colladaFiles -logs_dir /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/logs