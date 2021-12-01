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
        print(f"Skip {style}, {building}")
        continue
    style_per_building[building] = style


os.makedirs(args.out_dir, exist_ok=True)

duplicates_per_building = {}

for split in splits:
    total = 0
    train_file = os.path.join(args.out_dir, "trainfold{}.csv".format(split))
    test_file = os.path.join(args.out_dir, "testfold{}.csv".format(split))
    with open(train_file, "w") as f_train, open(test_file, "w") as f_test:
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
                            labels = np.zeros((len(styles)))
                            labels[styles.index(style)] = 1
                            np.save(file.replace(".npy", "_labels.npy"), labels)

                    if os.path.exists(file.replace(".npy", "_labels.npy")):
                        line = "{};{}\n".format(file, file.replace(".npy", "_labels.npy"))
                        total += 1
                        if any(f"/{b}/" in file for b in splits[split]['train_buildings']):
                            f_train.write(line)
                        elif any(f"/{b}/" in file for b in splits[split]['test_buildings']):
                            f_test.write(line)

    assert total == args.expected_total, f"Missing encoded file. Total: {total}, Expected: {args.expected_total}"
