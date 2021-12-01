import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dirs", required=True, type=str)
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--splits", required=True, type=str)
parser.add_argument("--styles", required=True, type=str)
parser.add_argument("--expected_total", required=True, type=int)
args = parser.parse_args()


styles = args.styles.split(",")
splits = json.load(open(args.splits, "r"))

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
                    model_name = os.path.basename(root)
                    file = os.path.join(root, _file)
                    if not os.path.exists(file.replace(".npy", "_labels.npy")):
                        style = [s for s in styles if s.lower() in file.lower()]
                        if len(style) < 1:
                            print(f"no style found for {file}")
                        else:
                            assert len(style) == 1
                            style = style[0]
                            labels = np.zeros((len(styles)))
                            labels[styles.index(style)] = 1
                            np.save(file.replace(".npy", "_labels.npy"), labels)

                    if os.path.exists(file.replace(".npy", "_labels.npy")):
                        line = "{};{}\n".format(file, file.replace(".npy", "_labels.npy"))
                        if model_name.split("_rot")[0] in splits[split]['train_buildings']:
                            total += 1
                            f_train.write(line)
                        elif model_name.split("_rot")[0] in splits[split]['test_buildings']:
                            total += 1
                            f_test.write(line)
                        else:
                            print(f"skip {line}")
                    else:
                        print(f"skip {file}")

    assert total == args.expected_total, f"Missing encoded files. Total: {total}, Expected: {args.expected_total}"
