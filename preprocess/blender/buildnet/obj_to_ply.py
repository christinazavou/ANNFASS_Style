import os
import bpy
import logging


LOGGER = logging.getLogger(name="obj_to_ply")


def set_logger_file(log_file):
    global LOGGER
    file_handler = logging.FileHandler(log_file, 'a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    for hdlr in LOGGER.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr, logging.FileHandler):
            LOGGER.removeHandler(hdlr)
    LOGGER.addHandler(file_handler)
    LOGGER.setLevel(logging.INFO)


def parse_buildings_csv(filename):
    buildings = []
    with open(filename, "r") as f:
        for line in f:
            buildings.append(line.strip().split(";")[1])
    LOGGER.info("buildings to process: {}".format(buildings))
    return buildings


def remove_objects(except_names=None):
    for objkey, objvalue in bpy.data.objects.items():
        if except_names is not None and objkey in except_names:
            continue
        else:
            bpy.data.objects.remove(objvalue, do_unlink=True)


def main(input_file, output_file):
    if os.path.exists(output_file):
        LOGGER.info("{} exists. Won't process..".format(output_file))
        return
    LOGGER.info("Processing {}".format(output_file))
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    remove_objects()
    bpy.ops.import_scene.obj(filepath=input_file)
    bpy.ops.export_mesh.ply(filepath=output_file, use_uv_coords=False, use_colors=False)
    bpy.ops.outliner.orphans_purge()


def run(in_dir, out_dir, buildings_csv):
    buildings = parse_buildings_csv(buildings_csv)
    for building in buildings:
        obj_file = os.path.join(in_dir, building, "{}.obj".format(building))
        ply_file = os.path.join(out_dir, building, "{}.ply".format(building))
        if os.path.exists(obj_file):
            main(obj_file, ply_file)
        else:
            LOGGER.warning("File {} doesnt exist".format(obj_file))


if __name__ == '__main__':

    import sys
    import argparse

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-ROOT', type=str, default="/media/christina/Data/ANNFASS_data/BUILDNET_Buildings")
        parser.add_argument('-OBJ_DIR', type=str, default="normalizedObj")
        parser.add_argument('-PLY_DIR', type=str, default="normalizedPly")
        parser.add_argument('-BUILDINGS_CSV', type=str, default="buildings.csv")
        parser.add_argument('-LOGS_DIR', type=str, default="/media/christina/Data/ANNFASS_data/BUILDNET_Buildings/logs")
        args = parser.parse_known_args(argv)[0]

    if not os.path.exists(args.LOGS_DIR):
        os.makedirs(args.LOGS_DIR)
    _log_file = os.path.join(args.LOGS_DIR, os.path.basename(args.BUILDINGS_CSV).replace('.csv', '.log'))
    set_logger_file(_log_file)

    LOGGER.info("Starting...")

    LOGGER.info("ROOT: {}".format(args.ROOT))
    LOGGER.info("OBJ_DIR: {}".format(args.OBJ_DIR))
    LOGGER.info("PLY_DIR: {}".format(args.PLY_DIR))
    LOGGER.info("BUILDINGS_CSV: {}".format(args.BUILDINGS_CSV))

    run(os.path.join(args.ROOT, args.OBJ_DIR),
        os.path.join(args.ROOT, args.PLY_DIR),
        os.path.join(args.ROOT, args.BUILDINGS_CSV))

    LOGGER.info("Ending...")
