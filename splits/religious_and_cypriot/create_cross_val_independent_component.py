import argparse
import json
import os
import random

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--buildings_csv", required=True, type=str)
parser.add_argument("--in_dir", required=True, type=str)
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--num_folds", default=10, type=int)
parser.add_argument("--styles", required=True, type=str)
parser.add_argument("--expected_buildings", required=True, type=int)
parser.add_argument("--elements", default="dome,tower,door,window,column", type=str)
# parser.add_argument("--expected_components", required=True, type=int)
args = parser.parse_args()

buildings_csv = args.buildings_csv
num_folds = args.num_folds
cross_val_json = f"{args.out_dir}/classification_cross_val_{num_folds}.json"
styles = args.styles.split(",")
elements = args.elements.split(",")


style_per_building = {}
buildings = []

buildings_df = pd.read_csv(buildings_csv, header=None, sep=';')
for idx, row in buildings_df.iterrows():
    style = row[0].lower()
    if style not in styles:
        style = 'unknown'
        print(f"{row[0]},{row[1]} --> unknown")
    building = row[1]
    style_per_building[building] = style
    buildings += [building]


models_per_style = {}
models = []
for building_dir in os.listdir(args.in_dir):
    if building_dir not in buildings:
        print(f"skip {building_dir}")
        continue
    style_init = style_per_building[building_dir]
    for component_dir in os.listdir(os.path.join(args.in_dir, building_dir)):
        element = [e for e in elements if e.lower() in component_dir.lower()]
        if len(element) == 0:
            print(f"skip {component_dir}")
            continue
        style = [s for s in styles if s.lower() in component_dir.lower()]
        if len(style) == 1:
            style = style[0]
        else:
            style = style_init
        if style != 'unknown':
            models_per_style.setdefault(style, [])
            models_per_style[style] += [f"{building_dir}/{component_dir}"]
            models += [f"{building_dir}/{component_dir}"]

buildings = set([m.split("/")[0] for m in models])
assert len(buildings) == args.expected_buildings, f"Missing building. Total: {len(buildings)}, Expected: {args.expected_buildings}"

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

