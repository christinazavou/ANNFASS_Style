import argparse
import os
import shlex
import subprocess
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=str, help="Encodings dir")
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--splits", required=True, type=str)
parser.add_argument("--py_exe", required=True, type=str)
parser.add_argument("--styles", default='gothic,byzantine,russian,baroque,asian', type=str)
parser.add_argument("--expected_total", required=True, type=int)
args = parser.parse_args()

run_configs = []


def get_run_configs():
    for split_iter_file in os.listdir(args.splits):
        assert split_iter_file.endswith(".json")
        split_file = join(args.splits, split_iter_file)
        split_iter = split_iter_file.replace('.json', '')
        for layer_dir in os.listdir(args.data_dir):
            for point_reduce_method_dir in os.listdir(os.path.join(args.data_dir, layer_dir)):
                dpath = join(args.data_dir, layer_dir, point_reduce_method_dir)
                if os.path.exists(dpath):
                    opath = join(args.out_dir, f"{layer_dir}_{point_reduce_method_dir}/{split_iter}")
                    run_configs.append(f"--data_dirs {dpath} "
                                       f"--out_dir {opath} "
                                       f"--splits {split_file} "
                                       f"--styles {args.styles} "
                                       f"--expected_total {args.expected_total}")


get_run_configs()


for i, config in enumerate(run_configs):
    cmd = "{} split_csv_generation.py {}".format(args.py_exe, config)
    print(cmd)
    proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

