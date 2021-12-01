import argparse
import json
import os
import random
import sys
import logging

dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dirname)
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


def run(root_dir, png_dirs, unique_dirs, split_root, splits_json, parts):

    unique_doesnt_exist = set()

    split = json.load(open(splits_json, "r"))

    train_buildings = split['train_buildings']
    val_buildings = split['test_buildings']
    test_buildings = split['val_buildings']

    train_file, test_file, val_file, all_file = make_output_files(split_root, out_dir)

    png_cnt = {}

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

        for png_dir, unique_dir in zip(png_dirs, unique_dirs):
            for root, folder, files in os.walk(os.path.join(root_dir, png_dir)):
                if not any(b in root for b in train_buildings+val_buildings+test_buildings):
                    continue
                for _file in files:
                    if _file.endswith(".png"):

                        if "_discard" in _file:
                            continue

                        file = os.path.join(root, _file)

                        building = [b for b in train_buildings+val_buildings+test_buildings if b in file]
                        if len(building) == 0:
                            continue
                        if len(building) > 1:
                            raise Exception()
                        building = building[0]
                        png_cnt.setdefault(building, {})

                        if unique_dir != "":  # we are running it at component level and we care for unique components
                            unique_building_dir = os.path.join(unique_dir, building)
                            if not os.path.exists(unique_building_dir):
                                unique_doesnt_exist.add(unique_building_dir)
                                continue
                            group = os.path.basename(root).replace("group_", "group")
                            png_cnt[building].setdefault(group, 0)

                            component_view = os.path.splitext(_file)[0].replace(building+"_", "").replace("style_mesh_", "")
                            component_view = f"{group}_{component_view}"
                            unique_components = os.listdir(unique_building_dir)
                            unique_components = [os.path.splitext(u)[0].replace("style_mesh_", "") for u in unique_components]

                            if not any(uc in component_view for uc in unique_components):
                                continue

                        if parts != "":  # we are running it at component level and we care for specific components
                            if not any(p.lower() in component_view.lower() for p in parts):
                                continue

                        png_cnt[building][group] += 1
                        current_new_line = "{}\n".format(file)
                        write_line(current_new_line)

    empty_png_count = 0
    non_empty_png_count = 0
    for building, building_dict in png_cnt.items():
        for group, group_cnt in building_dict.items():
            if group_cnt == 0:
                empty_png_count += 1
            else:
                non_empty_png_count +=1
    LOGGER.info("empty_png_cnt: {}".format(empty_png_count))
    print("empty_png_cnt: {}".format(empty_png_count))
    LOGGER.info("non_empty_png_cnt: {}".format(non_empty_png_count))
    print("non_empty_png_cnt: {}".format(non_empty_png_count))

    print(f"WARNING: {unique_doesnt_exist} doesnt exist")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--png_dirs", required=True, type=str)
    parser.add_argument("--unique_dirs", required=False, type=str, default=",", help="if provided it should be in corrsespondance to ply_dirs")
    parser.add_argument("--parts", required=False, type=str, default="", help="if provided it should be comma separated")
    parser.add_argument("--split_root", required=True, type=str)
    parser.add_argument("--splits_json", required=True, type=str)
    args = parser.parse_args()

    png_dirs = args.png_dirs.split(",")
    unique_dirs = args.unique_dirs.split(",")
    parts = args.parts.split(",")

    out_dir = ""
    for png_dir in png_dirs:
        out_dir += png_dir.split("/")[0] + "_" + os.path.basename(png_dir)
    if unique_dirs[0] != "":
        out_dir += "/unique"
    if len(parts) != 0:
        out_dir += "/"+"".join(parts)

    os.makedirs(os.path.join(args.split_root, out_dir), exist_ok=True)

    _log_file = os.path.join(args.split_root, out_dir, "png_splits_generation_unlabeled.log")
    LOGGER = set_logger_file(_log_file, LOGGER)
    LOGGER.info(args)

    run(args.root_dir, png_dirs, unique_dirs, args.split_root, args.splits_json, parts)
