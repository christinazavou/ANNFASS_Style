import argparse
import json
import os

import numpy as np
import pandas as pd

# todo: na ginei san to ply_splits_generation kai na svistei to annfass_content_style_splits.py

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, type=str)
parser.add_argument("--logs_dir", required=True, type=str)
parser.add_argument("--repo", default="BUILDNET_Buildings", type=str)
parser.add_argument("--content_dir", default="samplePoints/stylePly_cut10.0K_pgc_content512", type=str)
parser.add_argument("--style_dir", default="samplePoints/stylePly_cut10.0K_pgc_style4096", type=str)
parser.add_argument("--splits_dir", default="buildnet_content_style_splits", type=str)
parser.add_argument("--unique_dir", default="unique_point_clouds", type=str)
parser.add_argument("--buildings_csv", default="buildings_religious.csv", type=str)
parser.add_argument("--only_unique", default="True", type=str)
parser.add_argument("--train_pct", default=0.8, type=float)
parser.add_argument("--parts", default=None, type=str)
args = parser.parse_args()

parts = args.parts.split(",") if args.parts is not None else None

ply_content_dir = os.path.join(args.root_dir, args.repo, args.content_dir)
ply_style_dir = os.path.join(args.root_dir, args.repo, args.style_dir)
buildings_csv_path = os.path.join(args.root_dir, args.repo, args.buildings_csv)

df = pd.read_csv(buildings_csv_path, sep=";", header=None)
buildings = list(df[1].values)
train_buildings = int(args.train_pct*len(buildings))
train_buildings = np.random.randint(0, len(buildings), train_buildings)
train_buildings = [b for idx, b in enumerate(buildings) if idx in train_buildings]

content_dir = os.path.join(args.logs_dir, args.splits_dir, "content")
style_dir = os.path.join(args.logs_dir, args.splits_dir, "style")
os.makedirs(content_dir, exist_ok=True)
os.makedirs(style_dir, exist_ok=True)

content_train_file = os.path.join(content_dir, "train.txt")
content_test_file = os.path.join(content_dir, "test.txt")
style_train_file = os.path.join(style_dir, "train.txt")
style_test_file = os.path.join(style_dir, "test.txt")

log_file = os.path.join(args.logs_dir, args.splits_dir, "no_components_found.txt")

lf = open(log_file, "w")

with open(content_train_file, "w") as f_content_train, \
    open(content_test_file, "w") as f_content_test,\
    open(style_train_file, "w") as f_style_train,\
    open(style_test_file, "w") as f_style_test:

    for building in buildings:
        unique_file = os.path.join(args.root_dir, args.repo, args.unique_dir, building, "duplicates.json")
        if not os.path.exists(unique_file) and eval(args.only_unique):
            continue
        if not os.path.exists(os.path.join(ply_content_dir, building)):
            continue
        if not os.path.exists(os.path.join(ply_style_dir, building)):
            continue
        has_components = False
        if os.path.exists(unique_file):
            with open(unique_file, "r") as fin:
                unique_components = list(json.load(fin).keys())
        else:
            unique_components = None
        for component_file in os.listdir(os.path.join(ply_content_dir, building)):
            if component_file not in os.listdir(os.path.join(ply_style_dir, building)):
                continue  # only in content dir so skip
            if not any(unique in component_file for unique in unique_components) and eval(args.only_unique):
                continue  # skip duplicates
            if parts is not None and not any(p.lower() in component_file.lower() for p in parts):
                continue  # skip this part
            has_components = True
            assert component_file.endswith(".ply")
            component_name = component_file.replace(building+"_style_mesh_", "")
            if building in train_buildings:
                filepath = os.path.join(ply_content_dir, building, component_file)
                newline = "{};{};{}\n".format(filepath, building, component_name)
                f_content_train.write(newline)
                filepath = os.path.join(ply_style_dir, building, component_file)
                newline = "{};{};{}\n".format(filepath, building, component_name)
                f_style_train.write(newline)
            else:
                filepath = os.path.join(ply_content_dir, building, component_file)
                newline = "{};{};{}\n".format(filepath, building, component_name)
                f_content_test.write(newline)
                filepath = os.path.join(ply_style_dir, building, component_file)
                newline = "{};{};{}\n".format(filepath, building, component_name)
                f_style_test.write(newline)
        if not has_components:
            lf.write(building+"\n")
lf.close()
