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
    split_dir1 = os.path.join(split_root, sub_dir, "split_train_test")
    split_dir2 = os.path.join(split_root, sub_dir, "split_test")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(split_dir1, exist_ok=True)
    os.makedirs(split_dir2, exist_ok=True)

    train_file = os.path.join(split_dir, "train.txt")
    train_file1 = os.path.join(split_dir1, "train.txt")
    train_file2 = os.path.join(split_dir2, "train.txt")
    val_file = os.path.join(split_dir, "val.txt")
    val_file1 = os.path.join(split_dir1, "val.txt")
    val_file2 = os.path.join(split_dir2, "val.txt")
    test_file = os.path.join(split_dir, "test.txt")
    test_file1 = os.path.join(split_dir1, "test.txt")
    test_file2 = os.path.join(split_dir2, "test.txt")
    return train_file, train_file1, train_file2, test_file, test_file1, test_file2, val_file, val_file1, val_file2


def run(root_dir, ply_dirs, unique_dirs, split_root, splits_json, parts, num_folds=None):

    folds = json.load(open(splits_json, "r"))
    for fold_id, fold in folds.items():
        if num_folds is not None and int(fold_id) >= num_folds:
            break

        train_buildings = fold['train_buildings']
        test_buildings = fold['test_buildings']
        val_cnt = int(len(train_buildings) * 0.3)
        val_buildings = random.sample(train_buildings, val_cnt)
        train_buildings = [b for b in train_buildings if b not in val_buildings]

        train_file, train_file1, train_file2, test_file, test_file1, test_file2, val_file, val_file1, val_file2 = \
            make_output_files(split_root, out_dir+"/fold"+fold_id)

        empty_ply_cnt = 0
        non_empty_ply_cnt = 0

        with open(train_file, "w") as f_train, open(val_file, "w") as f_val, open(test_file, "w") as f_test, \
                open(train_file1, "w") as f_train1, open(val_file1, "w") as f_val1, open(test_file1, "w") as f_test1, \
                open(train_file2, "w") as f_train2, open(val_file2, "w") as f_val2, open(test_file2, "w") as f_test2:

            def write_line(new_line):
                in_training = any(b in new_line for b in train_buildings)
                in_val = any(b in new_line for b in val_buildings)
                in_test = any(b in new_line for b in test_buildings)
                if in_training:
                    f_train.write(new_line), f_train1.write(new_line), f_test2.write(new_line)
                elif in_val:
                    f_val.write(new_line), f_train1.write(new_line), f_test2.write(new_line)
                elif in_test:
                    f_test.write(new_line), f_test1.write(new_line), f_test2.write(new_line)
                else:
                    LOGGER.info(f'skip {new_line}')
                    print(f'skip {new_line}')

            for ply_dir, unique_dir in zip(ply_dirs, unique_dirs):
                for root, folder, files in os.walk(os.path.join(root_dir, ply_dir)):
                    if not any(b in root for b in train_buildings+val_buildings+test_buildings):
                        continue
                    for _file in files:
                        if _file.endswith(".ply"):

                            file = os.path.join(root, _file)

                            building = [b for b in train_buildings+val_buildings+test_buildings if b in file]
                            assert len(building) == 1
                            building = building[0]
                            if unique_dir != "":  # we are running it at component level and we care for unique components
                                unique_building_dir = os.path.join(root_dir, unique_dir, building)
                                component = os.path.splitext(_file)[0].replace(building+"_", "").replace("style_mesh_", "")
                                unique_components = os.listdir(unique_building_dir)
                                unique_components = [os.path.splitext(u)[0].replace("style_mesh_", "") for u in unique_components]
                                if component not in unique_components:
                                    continue

                            if parts != "":  # we are running it at component level and we care for specific components
                                component = os.path.splitext(_file)[0].replace(building+"_", "").replace("style_mesh_", "")
                                if not any(p.lower() in component.lower() for p in parts):
                                    continue

                            vertices, normals, labels = read_ply(ply_fn=file)
                            if len(vertices) == 0:
                                empty_ply_cnt += 1
                                continue
                            non_empty_ply_cnt += 1

                            current_new_line = "{}\n".format(file)
                            write_line(current_new_line)

        LOGGER.info("empty_ply_cnt: {}".format(empty_ply_cnt))
        print("empty_ply_cnt: {}".format(empty_ply_cnt))
        LOGGER.info("non_empty_ply_cnt: {}".format(non_empty_ply_cnt))
        print("non_empty_ply_cnt: {}".format(non_empty_ply_cnt))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--ply_dirs", required=True, type=str)
    parser.add_argument("--unique_dirs", required=False, type=str, default=",", help="if provided it should be in corrsespondance to ply_dirs")
    parser.add_argument("--parts", required=False, type=str, default="", help="if provided it should be comma separated")
    parser.add_argument("--split_root", required=True, type=str)
    parser.add_argument("--splits_json", required=True, type=str)
    parser.add_argument("--num_folds", required=False, type=int, default=1)
    args = parser.parse_args()

    ply_dirs = args.ply_dirs.split(",")
    unique_dirs = args.unique_dirs.split(",")
    parts = args.parts.split(",")

    out_dir = ""
    for ply_dir in ply_dirs:
        out_dir += ply_dir.split("/")[0] + "_" + os.path.basename(ply_dir)
    if unique_dirs[0] != "":
        out_dir += "/unique"
    if len(parts) != 0:
        out_dir += "/"+"".join(parts)

    os.makedirs(os.path.join(args.split_root, out_dir), exist_ok=True)

    _log_file = os.path.join(args.split_root, out_dir, "ply_splits_generation.log")
    LOGGER = set_logger_file(_log_file, LOGGER)
    LOGGER.info(args)

    run(args.root_dir, ply_dirs, unique_dirs, args.split_root, args.splits_json, parts, args.num_folds)
