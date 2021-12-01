import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import STYLES, parse_components_with_style_csv


parser = argparse.ArgumentParser()
parser.add_argument("--data_dirs", required=True, type=str)
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--splits", required=True, type=str)
parser.add_argument("--components_csv", required=True, type=str)
parser.add_argument("--override_labels", default="False", type=str)
parser.add_argument("--mode", default="encodings", type=str, help="images or encodings")
args = parser.parse_args()


components = parse_components_with_style_csv(args.components_csv)

splits = json.load(open(args.splits, "r"))

os.makedirs(args.out_dir, exist_ok=True)


def process_building_directory_encodings():

    building_components = [(s, b, c) for (s, b, c) in components if b in building_dir]
    if len(building_components) == 0:
        return

    building_components = [(s, b, c.replace("style_mesh_", "")) for (s, b, c) in building_components]
    building_components = [(s, b, c.split(".")[0]) for (s, b, c) in building_components]

    for component_file in os.listdir(os.path.join(data_dir, building_dir)):
        if not ".npy" in component_file:
            continue
        if "_labels.npy" in component_file:
            continue

        file = os.path.join(data_dir, building_dir, component_file)

        if not os.path.exists(file.replace(".npy", "_labels.npy")) or eval(args.override_labels):
            style = [s for (s, b, c) in building_components if c in component_file]
            if len(style) == 0:
                continue
            assert len(style) == 1
            style = style[0]
            labels = np.zeros((len(STYLES)))
            labels[STYLES.index(style)] = 1
            np.save(file.replace(".npy", "_labels.npy"), labels)

        if os.path.exists(file.replace(".npy", "_labels.npy")):
            line = "{};{}\n".format(file, file.replace(".npy", "_labels.npy"))
            if any(b in file for b in splits[split]['train_buildings']):
                f_train.write(line)
            elif any(b in file for b in splits[split]['test_buildings']):
                f_test.write(line)
            else:
                print(f"skip {file}")


def process_building_directory_images():

    building_components = [(s, b, c) for (s, b, c) in components if b in building_dir]
    if len(building_components) == 0:
        return

    building_components = [(s, b, c.replace("style_mesh_", "")) for (s, b, c) in building_components]
    building_components = [(s, b, c.split(".")[0]) for (s, b, c) in building_components]

    for group_dir in os.listdir(os.path.join(data_dir, building_dir)):
        group = group_dir.replace("_", "")
        for component_file in os.listdir(os.path.join(data_dir, building_dir, group_dir)):
            component = f"{group}_{component_file}"
            if not ".png" in component_file:
                continue
            if "_labels.npy" in component_file:
                continue

            file = os.path.join(data_dir, building_dir, group_dir, component_file)

            if not os.path.exists(file.replace(".png", "_labels.npy")) or eval(args.override_labels):
                style = [s for (s, b, c) in building_components if c in component]
                if len(style) == 0:
                    continue
                assert len(style) == 1
                style = style[0]
                labels = np.zeros((len(STYLES)))
                labels[STYLES.index(style)] = 1
                np.save(file.replace(".png", "_labels.npy"), labels)

            if os.path.exists(file.replace(".png", "_labels.npy")):
                line = "{};{}\n".format(file, file.replace(".png", "_labels.npy"))
                if any(b in file for b in splits[split]['train_buildings']):
                    f_train.write(line)
                elif any(b in file for b in splits[split]['test_buildings']):
                    f_test.write(line)
                else:
                    print(f"skip {file}")


if args.mode == 'encodings':
    process_building_directory = process_building_directory_encodings
else:
    assert args.mode == 'images'
    process_building_directory = process_building_directory_images


for split in splits:
    train_file = os.path.join(args.out_dir, "trainfold{}.csv".format(split))
    test_file = os.path.join(args.out_dir, "testfold{}.csv".format(split))
    with open(train_file, "w") as f_train, open(test_file, "w") as f_test:
        for data_dir in args.data_dirs.split(","):
            for building_dir in os.listdir(data_dir):
                process_building_directory()
