import argparse
import json
import os
import random

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--buildings_csv", required=True, type=str)
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--num_folds", default=10, type=int)
parser.add_argument("--styles", required=True, type=str)
parser.add_argument("--expected_total", required=True, type=int)
args = parser.parse_args()

buildings_csv = args.buildings_csv
num_folds = args.num_folds
cross_val_json = f"{args.out_dir}/classification_cross_val_{num_folds}.json"
styles = args.styles.split(",")


models_per_style = {}
models = []

buildings_df = pd.read_csv(buildings_csv, header=None, sep=';')
for idx, row in buildings_df.iterrows():
    style = row[0].lower()
    if style not in styles:
        print(f"skip {row[0]},{row[1]}")
        continue
    building = row[1]
    models_per_style.setdefault(style, [])
    models_per_style[style] += [building]
    models += [building]


assert len(models) == args.expected_total, f"Missing model. Total: {len(models)}, Expected: {args.expected_total}"


folds = {}
for style in styles:
    print(f"STYLE {style} has {len(models_per_style[style])} models.")
    random.shuffle(models_per_style[style])
    len_style = len(models_per_style[style])
    len_style_in_fold = len_style // num_folds
    for fold in range(num_folds):
        folds.setdefault(fold, [])
        folds[fold] += models_per_style[style][fold * len_style_in_fold:fold * len_style_in_fold + len_style_in_fold]

splits = {}
for fold_id, fold_data in folds.items():
    splits[fold_id] = {'train_buildings': [], 'test_buildings': []}
    for building in models:
        if building in fold_data:
            splits[fold_id]['test_buildings'] += [building]
        else:
            splits[fold_id]['train_buildings'] += [building]


os.makedirs(os.path.dirname(cross_val_json), exist_ok=True)
with open(cross_val_json, "w") as fout:
    json.dump(splits, fout, indent=2)
