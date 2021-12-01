import argparse
import os
import shlex
import subprocess
import sys
import time
from os.path import join

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.multiprocessing_utils import run_function_in_parallel, log_process_time


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--models_dir", required=True, type=str)
parser.add_argument("--svm_impl", default="simple", type=str)
parser.add_argument("--num_processes", default=4, type=int)
parser.add_argument("--py_exe", required=True, type=str)
parser.add_argument("--classes", type=str, default="")
parser.add_argument("--layers", type=str,)
args = parser.parse_args()

layers = args.layers.split(",")

run_configs = []
for data_dir in os.listdir(args.data_dir):
    if layers != [""] and data_dir not in layers:
        continue
    for experiment in os.listdir(join(args.data_dir, data_dir)):

        dpath = join(args.data_dir, data_dir, experiment)
        mpath = join(args.models_dir, data_dir, experiment)
        lpath = join(args.models_dir, data_dir, experiment, "log.txt")

        if not os.path.exists(mpath) or len(os.listdir(mpath)) == 0:
            run_configs.append((f"--data_dir {dpath} "
                                f"--models_dir {mpath} "
                                f"--svm_impl {args.svm_impl} "
                                f"--classes {args.classes}", lpath))
        else:
            print(f"{mpath} exists")


# print(run_configs)


def run_svm(configs, process_id):
    t_start_proc = time.time()

    for i, (config, logpath) in enumerate(configs):

        cmd = "{} run_svm.py {}".format(args.py_exe, config)
        print(cmd)
        proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if err != b"":
            print(err)
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        with open(logpath, "w") as fout:
            fout.write(out.decode("utf-8"))

        print("Process {}: Processed files ({}/{})".format(process_id, i + 1, len(configs)))

    log_process_time(process_id, t_start_proc)


# Preprocess models
t1 = time.time()
run_function_in_parallel(run_svm, args.num_processes, run_configs,)
# run_svm(run_configs[0:1], 0)
log_process_time("all", t1)


for data_dir in os.listdir(args.data_dir):
    if layers != [""] and data_dir not in layers:
        continue
    scores = []
    for experiment in os.listdir(join(args.models_dir, data_dir)):
        avg_f1_score_fn = join(args.models_dir, data_dir, experiment, "avg_f1_score.txt")
        with open(avg_f1_score_fn, "r") as fin:
            score = fin.readline().split(": ")[1]
            scores.append(float(score))
    avg_f1_score_fn = join(args.models_dir, data_dir, "overall_f1_score.txt")
    score = sum(scores) / len(scores)
    with open(avg_f1_score_fn, "w") as fout:
        fout.write(f"{np.round(np.mean(scores), 3)}\n{np.round(np.std(scores), 3)}")
