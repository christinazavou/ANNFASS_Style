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
parser.add_argument("--components_csv", required=False, type=str, default=None, help="Provide the styles in csv")
parser.add_argument("--part_segmentation_based", required=False, default='False', type=str)
parser.add_argument("--layer", type=str, default=None)
parser.add_argument("--point_reduce_method", type=str, default=None)
parser.add_argument("--component_reduce_method", type=str, default=None)
parser.add_argument("--override_labels", type=str, default="False")
args = parser.parse_args()

run_configs = []

if args.layer is None:
    LAYERS = ['layer_n-1_features', 'layer_n-2_features', 'layer_n-3_features',
              'discr_all', 'gen_enc_all', 'style_enc_all',
              'feature_concat', 'feature_bilinear', 'convd0_nearinterp',
              'z_dim_all', "z_dim_max", "z_dim_avg"]
else:
    LAYERS = [args.layer]
if args.point_reduce_method is None:
    POINT_REDUCE_METHODS = ['max', 'weighted_sum', 'avg', 'as_is']
else:
    POINT_REDUCE_METHODS = [args.point_reduce_method]
if args.component_reduce_method is None:
    COMPONENT_REDUCE_METHODS = ['avg', 'max', 'sum']
else:
    COMPONENT_REDUCE_METHODS = [args.component_reduce_method]


def get_run_part_segmentation_based_configs():
    for layer in LAYERS:
        for point_reduce_method in POINT_REDUCE_METHODS:
            for component_reduce_method in COMPONENT_REDUCE_METHODS:
                dpath = join(args.data_dir, layer, point_reduce_method+"_per_component", component_reduce_method)
                if os.path.exists(dpath):
                    opath = join(args.out_dir, "{}_{}_{}".format(layer, point_reduce_method+"_per_component", component_reduce_method))
                    run_configs.append(f"--data_dirs {dpath} "
                                       f"--out_dir {opath} "
                                       f"--splits {args.splits} "
                                       f"--override_labels {args.override_labels} "
                                       f"--components_csv {args.components_csv} ")
                dpath = join(args.data_dir, layer, point_reduce_method+"_per_component_rnv", component_reduce_method)
                if os.path.exists(dpath):
                    opath = join(args.out_dir, "{}_{}_{}".format(layer, point_reduce_method+"_per_component_rnv", component_reduce_method))
                    run_configs.append(f"--data_dirs {dpath} "
                                       f"--out_dir {opath} "
                                       f"--splits {args.splits} "
                                       f"--override_labels {args.override_labels} "
                                       f"--components_csv {args.components_csv} ")


def get_run_as_is_configs():
    for layer in LAYERS:
        for point_reduce_method in POINT_REDUCE_METHODS:
            dpath = os.path.join(args.data_dir, layer, point_reduce_method)
            if os.path.exists(dpath):
                opath = os.path.join(args.out_dir, "{}_{}".format(layer, point_reduce_method))
                run_configs.append(f"--data_dirs {dpath} "
                                   f"--out_dir {opath} "
                                   f"--splits {args.splits} "
                                   f"--override_labels {args.override_labels} "
                                   f"--components_csv {args.components_csv} ")


if eval(args.part_segmentation_based):
    get_run_part_segmentation_based_configs()
else:
    get_run_as_is_configs()


for i, config in enumerate(run_configs):
    cmd = "{} split_csv_generation.py {}".format(args.py_exe, config)
    print(cmd)
    proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

