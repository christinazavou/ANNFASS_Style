import argparse
import json
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--obj_dir", required=True, type=str)
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--num_folds", default=10, type=int)
parser.add_argument("--styles", required=True, type=str)
parser.add_argument("--label_file", default=None, type=str)
parser.add_argument("--expected_total", required=True, type=int)
args = parser.parse_args()

obj_dir = args.obj_dir
num_folds = args.num_folds
cross_val_json = f"{args.out_dir}/classification_cross_val_{num_folds}.json"
styles = args.styles.split(",")


models_per_style = {}
models = []

if args.label_file is not None and args.label_file != "":
    with open(args.label_file, "r") as fin:
        for line in fin.readlines():
            building_file, style = line.rstrip().split(" ")
            if int(style) < len(styles):
                models_per_style.setdefault(styles[int(style)], [])
                models_per_style[styles[int(style)]] += [building_file.replace(".ply", "")]
                models += [building_file.replace(".ply", "")]
else:
    for style_dir in os.listdir(obj_dir):
        if os.path.isdir(os.path.join(obj_dir, style_dir)):
            assert style_dir in styles, f"{style_dir} not in styles"
            models_per_style.setdefault(style_dir, [])
            for file in os.listdir(os.path.join(obj_dir, style_dir)):
                models_per_style[style_dir] += [file.replace(".ply", "")]
                models += [file.replace(".ply", "")]


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
