import argparse
import json
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--buildings_csv", required=True, type=str)
parser.add_argument("--data_dirs", required=True, type=str)
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--splits", required=True, type=str)
parser.add_argument("--styles", required=True, type=str)
parser.add_argument("--expected_total", required=True, type=int)
args = parser.parse_args()


styles = args.styles.split(",")
splits = json.load(open(args.splits, "r"))

buildings_df = pd.read_csv(args.buildings_csv, header=None, sep=';')
style_per_building = {}
for idx, row in buildings_df.iterrows():
    style = row[0].lower()
    building = row[1]
    if style not in styles:
        style = 'unknown'
        print(f"{style}, {building} --> unknown")
    style_per_building[building] = style


os.makedirs(args.out_dir, exist_ok=True)

duplicates_per_building = {}

for split in splits:
    total = 0
    train_file = os.path.join(args.out_dir, "trainfold{}.csv".format(split))
    test_file = os.path.join(args.out_dir, "testfold{}.csv".format(split))
    with open(train_file, "w") as f_train, open(test_file, "w") as f_test:
        train_buildings = splits[split]['train_buildings']
        test_buildings = splits[split]['test_buildings']
        train_buildings = [b.replace("style_mesh_", "") for b in train_buildings]
        test_buildings = [b.replace("style_mesh_", "") for b in test_buildings]
        for data_dir in args.data_dirs.split(","):
            for root, folder, files in os.walk(data_dir):
                for _file in files:
                    if not ".npy" in _file:
                        continue
                    if "_labels.npy" in _file:
                        continue
                    building = os.path.basename(root)
                    file = os.path.join(root, _file)
                    if not os.path.exists(file.replace(".npy", "_labels.npy")):
                        if building in style_per_building:
                            style = style_per_building[building]
                            if style == 'unknown':
                                style_component = [s for s in styles if s.lower() in component.lower()]
                                if len(style_component) == 1:
                                    style = style_component[0]
                            if style != "unknown":
                                labels = np.zeros((len(styles)))
                                labels[styles.index(style)] = 1
                                np.save(file.replace(".npy", "_labels.npy"), labels)

                    component = os.path.basename(file).replace(".npy", "")
                    building = building.replace("style_mesh_", "")
                    component = component.replace("style_mesh_", "")

                    if os.path.exists(file.replace(".npy", "_labels.npy")):
                        line = "{};{}\n".format(file, file.replace(".npy", "_labels.npy"))
                        if f"{building}/{component}" in train_buildings:
                            total += 1
                            f_train.write(line)
                        elif f"{building}/{component}" in test_buildings:
                            total += 1
                            f_test.write(line)
                        else:
                            print(f"missing {building}, {component}")
                    else:
                        print(f"missed {building}, {component}")

    assert total == args.expected_total, f"Missing encoded file. Total: {total}, Expected: {args.expected_total}"
