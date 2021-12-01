import argparse
import os
import shlex
import subprocess
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--splits", required=True, type=str)
parser.add_argument("--py_exe", required=True, type=str)
parser.add_argument("--styles", default='gothic,byzantine,russian,baroque,asian', type=str)
parser.add_argument("--label_file", default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/compressed_files/Data-all/Data/building/labels.txt", type=str)
parser.add_argument("--expected_total", required=True, type=int)
args = parser.parse_args()

run_configs = []


def get_run_configs():
    for layer_dir in os.listdir(args.data_dir):
        for point_reduce_method_dir in os.listdir(os.path.join(args.data_dir, layer_dir)):
            dpath = join(args.data_dir, layer_dir, point_reduce_method_dir)
            if os.path.exists(dpath):
                opath = join(args.out_dir, "{}_{}".format(layer_dir, point_reduce_method_dir))
                run_configs.append(f"--data_dirs {dpath} "
                                   f"--out_dir {opath} "
                                   f"--splits {args.splits} "
                                   f"--styles {args.styles} "
                                   f"--label_file {args.label_file} "
                                   f"--expected_total {args.expected_total}")

get_run_configs()


for i, config in enumerate(run_configs):
    cmd = "{} split_csv_generation.py {}".format(args.py_exe, config)
    print(cmd)
    proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

