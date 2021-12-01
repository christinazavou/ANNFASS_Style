import argparse
import os
import shlex
import subprocess
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.multiprocessing_utils import run_function_in_parallel, log_process_time


parser = argparse.ArgumentParser()
parser.add_argument("--encodings_dir", required=True, type=str)
parser.add_argument("--result_dir", required=True, type=str)
parser.add_argument("--splits_dir", required=True, type=str)
parser.add_argument("--splits", default=10, type=int)
parser.add_argument("--num_processes", default=4, type=int)
parser.add_argument("--py_exe", required=True, type=str)
parser.add_argument('--cuda', action='store_true', default=False,)
args = parser.parse_args()


run_configs = []

for layer_dir in os.listdir(args.encodings_dir):
    for method_dir in os.listdir(os.path.join(args.encodings_dir, layer_dir)):
        encodings_path = os.path.join(args.encodings_dir, layer_dir, method_dir)
        for root, dirs, files in os.walk(encodings_path):
            for file in files:
                if file.endswith(".npy"):
                    enc = np.load(os.path.join(root, file))
                    break
        new_config = f"--encodings_path {encodings_path} "+\
                     f"--result_dir {args.result_dir} "+\
                     f"--splits_dir {args.splits_dir} "+\
                     f"--splits {args.splits} "+\
                     f"--exp_name {layer_dir}_{method_dir} "+\
                     f"--num_input {enc.shape[1]}"
        if args.cuda:
            new_config += " --cuda"
        run_configs.append(new_config)


def run_triplet_embeddings(configs, process_id):
    t_start_proc = time.time()

    for i, config in enumerate(configs):
        cmd = "{} main_triplet_embeddings.py {}".format(args.py_exe, config)
        print(cmd)
        proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if err != b"":
            print(err)
        print("Process {}: Processed files ({}/{})".format(process_id, i + 1, len(configs)))

    log_process_time(process_id, t_start_proc)


# Preprocess models
t1 = time.time()
run_function_in_parallel(run_triplet_embeddings, args.num_processes, run_configs,)
log_process_time("all", t1)

# t1 = time.time()
# for i, config in enumerate(run_configs):
#
#     cmd = "{} main_triplet_embeddings.py {}".format(args.py_exe, config)
#     print(cmd)
#     proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     out, err = proc.communicate()
#
#     if err != b"":
#         print(err)
#
# log_process_time("all", t1)
