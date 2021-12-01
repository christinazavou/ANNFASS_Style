import argparse
import json
import os
import random
import sys
import logging

dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dirname)
from common.mesh_utils import read_ply
from common.utils import set_logger_file

LOGGER = logging.getLogger(__file__)


def make_output_files(split_root, sub_dir):
    split_dir = os.path.join(split_root, sub_dir, "split_train_val_test")
    os.makedirs(split_dir, exist_ok=True)

    train_file = os.path.join(split_dir, "train.txt")
    val_file = os.path.join(split_dir, "val.txt")
    test_file = os.path.join(split_dir, "test.txt")
    all_file = os.path.join(split_dir, "all.txt")
    return train_file, test_file, val_file, all_file


def run(root_dir, simple_obj_dirs, unique_dirs, split_root, splits_json, parts):

    unique_doesnt_exist = set()

    split = json.load(open(splits_json, "r"))

    train_buildings = split['train_buildings']
    val_buildings = split['test_buildings']
    test_buildings = split['val_buildings']

    train_file, test_file, val_file, all_file = make_output_files(split_root, out_dir)

    buildings_non_empty = set()

    with open(train_file, "w") as f_train, open(val_file, "w") as f_val, \
            open(test_file, "w") as f_test, open(all_file, "w") as f_all:

        def write_line(new_line):
            in_training = any(b in new_line for b in train_buildings)
            in_val = any(b in new_line for b in val_buildings)
            in_test = any(b in new_line for b in test_buildings)
            if in_training:
                f_train.write(new_line)
                f_all.write(new_line)
            elif in_val:
                f_val.write(new_line)
                f_all.write(new_line)
            elif in_test:
                f_test.write(new_line)
                f_all.write(new_line)
            else:
                LOGGER.info(f'skip {new_line}')
                print(f'skip {new_line}')

        for simple_obj_dir, unique_dir in zip(simple_obj_dirs, unique_dirs):
            for root, folder, files in os.walk(os.path.join(root_dir, simple_obj_dir)):
                if not any(b in root for b in train_buildings+val_buildings+test_buildings):
                    continue
                for _file in files:
                    if _file.endswith(".obj"):

                        file = os.path.join(root, _file)

                        building = [b for b in train_buildings+val_buildings+test_buildings if b in file]
                        if len(building) == 0:
                            continue
                        if len(building) > 1:
                            raise Exception()
                        building = building[0]
                        if unique_dir != "":  # we are running it at component level and we care for unique components
                            unique_building_dir = os.path.join(unique_dir, building)
                            if not os.path.exists(unique_building_dir):
                                unique_doesnt_exist.add(unique_building_dir)
                                continue
                            component = os.path.basename(os.path.dirname(file)).replace("style_mesh_", "")
                            unique_components = os.listdir(unique_building_dir)
                            unique_components = [os.path.splitext(u)[0].replace("style_mesh_", "") for u in unique_components]
                            if component not in unique_components:
                                continue

                        if parts != "":  # we are running it at component level and we care for specific components
                            component = os.path.basename(os.path.dirname(file)).replace("style_mesh_", "")
                            if not any(p.lower() in component.lower() for p in parts):
                                continue

                        buildings_non_empty.add(building)
                        current_new_line = "{}\n".format(file)
                        write_line(current_new_line)

    LOGGER.info("buildings_non_empty: {}".format(buildings_non_empty))
    print("buildings_non_empty: {}".format(buildings_non_empty))

    print(f"WARNING: {unique_doesnt_exist} doesnt exist")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--simple_obj_dirs", required=True, type=str)
    parser.add_argument("--unique_dirs", required=False, type=str, default=",", help="if provided it should be in corrsespondance to ply_dirs")
    parser.add_argument("--parts", required=False, type=str, default="", help="if provided it should be comma separated")
    parser.add_argument("--split_root", required=True, type=str)
    parser.add_argument("--splits_json", required=True, type=str)
    args = parser.parse_args()

    simple_obj_dirs = args.simple_obj_dirs.split(",")
    unique_dirs = args.unique_dirs.split(",")
    parts = args.parts.split(",")

    out_dir = ""
    for simple_obj_dir in simple_obj_dirs:
        out_dir += os.path.basename(simple_obj_dir)
    if unique_dirs[0] != "":
        out_dir += "/unique"
    if len(parts) != 0:
        out_dir += "/"+"".join(parts)

    os.makedirs(os.path.join(args.split_root, out_dir), exist_ok=True)

    _log_file = os.path.join(args.split_root, out_dir, "simple_obj_splits_generation_unlabeled.log")
    LOGGER = set_logger_file(_log_file, LOGGER)
    LOGGER.info(args)

    run(args.root_dir, simple_obj_dirs, unique_dirs, args.split_root, args.splits_json, parts)
