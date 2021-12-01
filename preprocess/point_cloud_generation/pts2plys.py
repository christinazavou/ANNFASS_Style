import argparse
import os
import sys
import time

from pts2ply import process_buildings

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.utils import str2bool, parse_buildings_csv
from common.multiprocessing_utils import run_function_in_parallel, log_process_time


parser = argparse.ArgumentParser()
parser.add_argument("--obj_dir", default="normalizedObj", type=str)
parser.add_argument("--pts_dir", default="samplePoints/point_cloud_10K", type=str)
parser.add_argument("--ply_dir_prefix", default="samplePoints/ply", type=str)
parser.add_argument("--groups_dir", default=None, type=str)
parser.add_argument("--buildings_csv", default="buildings.csv", type=str)
parser.add_argument("--cut_at", default=-1, type=int)
parser.add_argument("--per_component", default=False, type=str2bool)
parser.add_argument("--rnv", default="ridge_valley", type=str)
parser.add_argument("--color", default="colorPly", type=str)
parser.add_argument("--num_processes", default=4, type=int)
args = parser.parse_args()


ply_dir_suffix = ""
if args.cut_at == -1:
    ply_dir_suffix += "_nocut"
else:
    ply_dir_suffix += f"_cut{round(args.cut_at/1000, 1)}K"
if args.per_component:
    if args.groups_dir is None:
        ply_dir_suffix += "_pc"
    else:
        ply_dir_suffix += "_pgc"
else:
    if args.groups_dir is not None:
        ply_dir_suffix += "_wg"
pl_dir = os.path.join(args.ply_dir_prefix+ply_dir_suffix)
print(f"Generating output ply files in {pl_dir}")

rnvd = os.path.join(args.pts_dir.replace("point_cloud", args.rnv))
colord = os.path.join(args.pts_dir.replace("point_cloud", args.color))

buildings = parse_buildings_csv(args.buildings_csv)

data = []
for building in buildings:

    pt_file = os.path.join(args.pts_dir, "{}.pts".format(building))
    f_file = os.path.join(args.pts_dir.replace("point_cloud", "faces"), "{}.txt".format(building))

    if args.groups_dir is not None:
        g_file = os.path.join(args.groups_dir, building, "groups.json")
    else:
        g_file = None

    o_file = os.path.join(args.obj_dir, building, "{}.obj".format(building))
    if "style_mesh" in pt_file:
        o_file = os.path.join(args.obj_dir, building, "{}_style_mesh.obj".format(building))

    r_file = None
    if os.path.exists(rnvd):
        r_file = os.path.join(rnvd, "{}.txt".format(building))

    c_file = None
    if os.path.exists(colord):
        c_file = os.path.join(colord, "{}.ply".format(building))

    data.append((building, g_file, pt_file, f_file, o_file, r_file, c_file, pl_dir))

# Preprocess models
t1 = time.time()
run_function_in_parallel(process_buildings, args.num_processes, data,
                         cut_at=args.cut_at, per_component=args.per_component)
# process_buildings(data, 0, cut_at=args.cut_at, per_component=args.per_component)
log_process_time("all", t1)
