import argparse
import json
import os
import random


models_per_style = {}
models = []


def get_data():
    global models
    ignore = set()
    if args.exclude_file and os.path.exists(args.exclude_file):
        with open(args.exclude_file, "r") as fin:
            for line in fin.readlines():
                ignore.add(line.rstrip())
        print(f"Will ignore {len(ignore)} models")
    for building_dir in os.listdir(obj_dir):
        if ".obj" in building_dir:
            building_dir = building_dir.replace(".obj", "")
        style = [s for s in styles if s.lower() in building_dir.lower()]
        if len(style) < 1:
            print(f"style not found for f{building_dir}: {style}")
        else:
            style = style[0]
            models_per_style.setdefault(style, [])
            models_per_style[style] += [building_dir]
            if building_dir not in ignore:
                models += [building_dir]

    assert len(models) == args.expected_total, f"Missing model. Total: {len(models)}, Expected: {args.expected_total}"


def run():
    folds = {}
    for style in styles:
        print(f"STYLE {style} has {len(models_per_style[style])} models.")
        random.shuffle(models_per_style[style])
        len_style = len(models_per_style[style])
        len_style_in_fold = len_style // num_folds
        for fold in range(num_folds):
            folds.setdefault(fold, [])
            folds[fold] += models_per_style[style][
                           fold * len_style_in_fold:fold * len_style_in_fold + len_style_in_fold]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--num_folds", default=5, type=int)
    parser.add_argument("--repeat", default=5, type=int)
    parser.add_argument("--styles", required=True, type=str)
    parser.add_argument("--expected_total", required=True, type=int)
    parser.add_argument("--exclude_file", required=False, type=str)
    args = parser.parse_args()

    obj_dir = args.obj_dir
    num_folds = args.num_folds
    styles = args.styles.split(",")

    get_data()

    for repetition in range(args.repeat):
        cross_val_json = f"{args.out_dir}/classification_cross_val_{num_folds}/split_iter_{repetition}.json"
        run()
