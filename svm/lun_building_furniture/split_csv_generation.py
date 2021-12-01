import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dirs", default="/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/from_turing/sep9/trained_on_buildnet_buildings/original_clean/s8/encodings_building_lun/discr_all/max", type=str)
parser.add_argument("--out_dir", default="/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/lun_data/building/classification_cross_val_5_csv", type=str)
parser.add_argument("--splits", default='/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/lun_data/building/classification_cross_val_5.json', type=str)
parser.add_argument("--styles", default='gothic,byzantine,russian,baroque,asian', type=str)
parser.add_argument("--label_file", default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/compressed_files/Data-all/Data/building/labels.txt", type=str)
parser.add_argument("--expected_total", required=True, type=int)
args = parser.parse_args()


styles = args.styles.split(",")
splits = json.load(open(args.splits, "r"))

label_file = args.label_file
style_per_building = {}
with open(label_file, "r") as fin:
    for line in fin.readlines():
        building_file, style = line.rstrip().split(" ")
        if int(style) < len(styles):
            style_per_building[building_file.replace(".ply", "")] = styles[int(style)]

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
                    file = os.path.join(root, _file)
                    building = root.split("/")[-1]
                    if building not in style_per_building:
                        continue
                    if not os.path.exists(file.replace(".npy", "_labels.npy")):
                        style = style_per_building[building]
                        labels = np.zeros((len(styles)))
                        labels[styles.index(style)] = 1
                        np.save(file.replace(".npy", "_labels.npy"), labels)

                    if os.path.exists(file.replace(".npy", "_labels.npy")):
                        total += 1
                        line = "{};{}\n".format(file, file.replace(".npy", "_labels.npy"))
                        if any(f"/{b}/" in file for b in splits[split]['train_buildings']):
                            f_train.write(line)
                        elif any(f"/{b}/" in file for b in splits[split]['test_buildings']):
                            f_test.write(line)
                        else:
                            print(f"skip {file}")
                    else:
                        print(f"skip {file}")
    assert total == args.expected_total, f"Missing encoded file. Total: {total}, Expected: {args.expected_total}"
