import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.utils import parse_buildings_csv
from common.mesh_utils import read_ply, normalize_coords, write_ply
from preprocess.mesh_smoothing.fps import farthest_point_sampling


def smooth_component(ply_sfe_file, c_file, s_file, detail_samples, coarse_samples):

    if os.path.exists(c_file) and os.path.exists(s_file):
        return

    vertices_detail, _, _ = read_ply(ply_sfe_file)
    if len(vertices_detail) < detail_samples:
        return  # skip this component

    vertices_detail = normalize_coords(vertices_detail, "box")
    vertices_detail = vertices_detail[np.random.randint(vertices_detail.shape[0], size=detail_samples), :]

    coarse_ratio = round(coarse_samples / len(vertices_detail), 2) + 0.01  # to ensure more than needed
    vertices_coarse = farthest_point_sampling(vertices_detail, coarse_ratio)
    vertices_coarse = vertices_coarse[np.random.randint(vertices_coarse.shape[0], size=coarse_samples), :]

    if len(vertices_coarse) < coarse_samples:
        return  # skip this component

    os.makedirs(content_out_dir, exist_ok=True)  # here so that empty building folders won't be created
    os.makedirs(style_out_dir, exist_ok=True)
    write_ply(c_file, vertices_coarse)
    write_ply(s_file, vertices_detail[:detail_samples])


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, type=str)
parser.add_argument("--repo", default="ANNFASS_Buildings", type=str)
parser.add_argument("--ply_per_component_dir", default="samplePoints/ply_nocut_pgc", type=str)
parser.add_argument("--buildings_csv", default="buildings.csv", type=str)
parser.add_argument("--detail_samples", default=4096, type=int)
parser.add_argument("--coarse_samples", default=512, type=int)
args = parser.parse_args()

root_dir = os.path.join(args.root, args.repo)

buildings = parse_buildings_csv(os.path.join(root_dir, args.buildings_csv))

data = []
for building in buildings:

    ply_dir = os.path.join(root_dir, args.ply_per_component_dir, building)
    if not os.path.exists(ply_dir):
        continue

    content_out_dir = os.path.join(root_dir, f"{args.ply_per_component_dir}_content{args.coarse_samples}", building)
    style_out_dir = os.path.join(root_dir, f"{args.ply_per_component_dir}_style{args.detail_samples}", building)

    print(f"processing building {building}")
    for component_file in os.listdir(ply_dir):
        if os.path.isfile(os.path.join(ply_dir, component_file)) and component_file.endswith(".ply"):
            content_file = os.path.join(content_out_dir, component_file)
            style_file = os.path.join(style_out_dir, component_file)

            smooth_component(os.path.join(ply_dir, component_file),
                             content_file,
                             style_file,
                             args.detail_samples,
                             args.coarse_samples)

