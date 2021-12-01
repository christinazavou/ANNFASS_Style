import argparse
import os
import shlex
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.multiprocessing_utils import run_function_in_parallel, log_process_time


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--models_dir", required=True, type=str)
parser.add_argument("--svm_impl", default="SVM", type=str)
parser.add_argument("--num_processes", default=4, type=int)
parser.add_argument("--ignore_classes", type=str, default="")
parser.add_argument("--avg_f1_nan", type=str, default="np.nan", help="e.g. use np.nan or 0")
parser.add_argument("--py_exe", required=True, type=str)
parser.add_argument("--unique_dirs", type=str)
parser.add_argument("--override", type=str, default=False)
args = parser.parse_args()

run_configs = []
for data_dir in os.listdir(args.data_dir):
    dpath = os.path.join(args.data_dir, data_dir)
    mpath = os.path.join(args.models_dir, data_dir)
    if not os.path.exists(mpath) or len(os.listdir(mpath)) == 0 or eval(args.override):
        run_configs.append(f"--data_dir {dpath} "
                           f"--models_dir {mpath} "
                           f"--svm_impl {args.svm_impl} "
                           f"--ignore_classes {args.ignore_classes} "
                           f"--avg_f1_nan {args.avg_f1_nan} "
                           f"--unique_dirs {args.unique_dirs}")
    else:
        print(f"{mpath} exists")

# print(run_configs)


def run_svm(configs, process_id):
    t_start_proc = time.time()

    for i, config in enumerate(configs):

        cmd = "{} run_svm.py {}".format(args.py_exe, config)
        print(cmd)
        proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if err != b"":
            print(err)
        print("Process {}: Processed files ({}/{})".format(process_id, i + 1, len(configs)))

    log_process_time(process_id, t_start_proc)


# Preprocess models
t1 = time.time()
run_function_in_parallel(run_svm, args.num_processes, run_configs,)
log_process_time("all", t1)
