import argparse
import os
import random
import sys
import time

import numpy as np
from plyfile import PlyData

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.mesh_utils import read_pts, read_face_indices
from common.multiprocessing_utils import log_process_time, run_function_in_parallel
from common.utils import parse_buildings_csv


def process_one_ply(in_ply, out_ply, samples, rnd=True):
    plydata = PlyData.read(in_ply)
    data = plydata.elements[0].data
    if rnd:
        sample_indices = random.sample(list(range(data.shape[0])), samples)
        data = data[sample_indices]
    else:
        data = data[:samples]
    plydata.elements[0].data = data
    plydata.write(out_ply)


def process_multiple_plys(configs, process_id):
    for config in configs:
        process_one_ply(*config)
    print("Process {}: Processed {} files.".format(process_id, len(configs)))


def run_ply(root_dir, out_dir, num_processes, samples, override, rnd):
    os.makedirs(out_dir, exist_ok=True)
    configs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ply"):
                in_ply = os.path.join(root, file)
                out_ply = os.path.join(out_dir, file)
                if buildings is not None:
                    if not any(b.lower() in in_ply.lower() for b in buildings):
                        continue
                if not os.path.exists(out_ply) or override:
                    configs.append((in_ply, out_ply, samples, rnd))
    print("will process {} files with {} processes".format(len(configs), num_processes))
    t1 = time.time()
    run_function_in_parallel(process_multiple_plys, num_processes, configs,)
    log_process_time("all", t1)


def run_pts(input_cloud_dir, output_cloud_dir, samples):
    os.makedirs(output_cloud_dir, exist_ok=True)
    os.makedirs(output_cloud_dir.replace("point_cloud", "faces"), exist_ok=True)
    for root, dirs, files in os.walk(input_cloud_dir):
        for file in files:
            if file.endswith(".pts"):
                print("Processing {}".format(file))

                pts_file = os.path.join(root, file)
                faces_file = pts_file.replace("point_cloud", "faces").replace(".pts", ".txt")
                pts_out_f = os.path.join(output_cloud_dir, file)
                faces_out_f = pts_out_f.replace("point_cloud", "faces").replace(".pts", ".txt")

                coords, normals = read_pts(pts_file)
                face_indices = read_face_indices(faces_file)

                sample_indices = random.sample(list(range(len(coords))), samples)
                coords = coords[sample_indices, :]
                normals = normals[sample_indices, :]
                face_indices = np.array(face_indices)[sample_indices]

                with open(pts_out_f, "w") as ptsf:
                    for coord, normal in zip(coords, normals):
                        ptsf.write("{} {} {} {} {} {}\n".format(coord[0], coord[1], coord[2],
                                                                normal[0], normal[1], normal[2]))

                with open(faces_out_f, "w") as ff:
                    for fidx in face_indices:
                        ff.write("{}\n".format(fidx))


parser = argparse.ArgumentParser()
parser.add_argument("--input_cloud_dir", required=True, type=str)
parser.add_argument("--output_cloud_dir", required=True, type=str)
parser.add_argument("--samples", default=30000, type=int)
parser.add_argument("--mode", default="ply", type=str, help="ply or pts")
parser.add_argument("--num_processes", default=1, type=int, )
parser.add_argument("--override", default="False", type=str, )
parser.add_argument("--building_csv", default="", type=str, )
parser.add_argument("--random", default="True", type=str, help="be careful..if you need faces use False")
args = parser.parse_args()


if args.building_csv:
    buildings = parse_buildings_csv(args.building_csv)
else:
    buildings = None


if "ply" in args.output_cloud_dir or args.mode == "ply":
    run_ply(args.input_cloud_dir, args.output_cloud_dir,
            args.num_processes, args.samples, eval(args.override), eval(args.random))
else:
    run_pts(args.input_cloud_dir, args.output_cloud_dir, args.samples)
