import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from preprocess.blender.scene_utils import cleanup
from preprocess.blender.mesh_utils import triangulate_all
from preprocess.blender.io_utils import load_obj, bpy


LOGGER = logging.getLogger(name="triangulate")


def process_model(input_file, output_file):
    if os.path.exists(output_file):
        LOGGER.info("Output file exists. Won't process..")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleanup()
    LOGGER.info("loading...")
    load_obj(input_file)
    LOGGER.info("triangulating...")
    triangulate_all()
    LOGGER.info("unit scaling...")
    LOGGER.info("saving...")
    bpy.ops.export_scene.obj(filepath=output_file, axis_up='Y', axis_forward='-Z')


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-inp_dir', type=str, required=True)
        parser.add_argument('-out_dir', type=str, required=True)
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    LOGGER.info("Starting...")

    for model_dir in os.listdir(args.inp_dir):
        in_file = os.path.join(args.inp_dir, model_dir, "model.obj")
        out_file = os.path.join(args.out_dir, model_dir, "model.obj")
        if not os.path.exists(out_file):
            try:
                process_model(in_file, out_file)
            except:
                print(f"couldnt do {model_dir}")

    LOGGER.info("Ending...")


# blender --background --python triangulate.py -- -inp_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_HU_YU_LUN_BUILDNET/chair_yu -out_dir /media/graphicslab/BigData1/zavou/ANNFASS_DATA/DATA_HU_YU_LUN_BUILDNET/chair_yu_tri
