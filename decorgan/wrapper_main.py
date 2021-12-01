import argparse
import os
import shlex
import subprocess
import sys
import time
import torch
import multiprocessing as mp


def run_function_in_parallel(func, n_processes, data, **kwargs):
    """
    :param func the function to be called by multiple parallel processes
    :param n_processes how many processes
    :param data a list of the data to be chunked and passed in multiple calls of func
    :param args extra parameters to be passed in func
    e.g. if func is defined as func(d, p1, p2) then args is (p1, p2)
         if func is defined as func(d, p1=0, p2=0) then args can be (p1, p2) or {p1:0, p2:0}
    """
    step = int(round(len(data) / float(n_processes)))
    processes = []
    for process_id in range(n_processes):
        start_idx = process_id * step
        if process_id == n_processes - 1:
            end_idx = len(data)
        else:
            end_idx = (process_id + 1) * step
        data_chunk = data[start_idx:end_idx]
        if len(data_chunk) == 0:
            continue
        t = mp.Process(target=func, args=(data_chunk, process_id), kwargs=kwargs)
        processes.append(t)
        t.start()

    # Wait for processes to end
    for oneProc in processes:
        oneProc.join()


def log_process_time(process_id, start_time):
    elapsed_time = time.time() - start_time
    print("Terminating process {process_id}. Time passed: {hours:d}:{minutes:d}:{seconds:d}"
          .format(process_id=process_id,
                  hours=int((elapsed_time / 60 ** 2) % (60 ** 2)),
                  minutes=int((elapsed_time / 60) % (60)),
                  seconds=int(elapsed_time % 60)))


parser = argparse.ArgumentParser()
parser.add_argument("--py_exe", dest="py_exe", help="python executable")
parser.add_argument("--config_files", dest="config_files", help="comma separated")
FLAGS = parser.parse_args()

print("started wrapper_main")
num_devices = torch.cuda.device_count()
print("found {} devices".format(num_devices))
cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
devices = []
for device_num in range(num_devices):
    if str(device_num) in cuda_visible_devices.split(","):
        devices.append(device_num)
config_files = FLAGS.config_files.split(",")
config_files = config_files[:len(devices)]

run_configs = []
for device, config_file in zip(devices, config_files):
    run_configs.append((f"--config_yml {config_file} --gpu {device}",
                        config_file.replace('.yml', '.out'),
                        config_file.replace('.yml', '.err')))

print(run_configs)


def run_mymain(configs, process_id):
    t_start_proc = time.time()

    for i, (config, out_file, err_file) in enumerate(configs):

        cmd = f"{FLAGS.py_exe} mymain.py {config}"
        print(cmd)
        proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        with open(err_file, "wb") as fout:
            fout.write(err)
        with open(out_file, "wb") as fout:
            fout.write(out)

        print("Process {}: Processed files ({}/{})".format(process_id, i + 1, len(configs)))

    log_process_time(process_id, t_start_proc)


# Preprocess models
t1 = time.time()
run_function_in_parallel(run_mymain, len(devices), run_configs,)
log_process_time("all", t1)

print("finished wrapper_main")
