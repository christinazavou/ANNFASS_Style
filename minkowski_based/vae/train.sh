#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=titanx-long    # Partition to submit to
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=10000   # Memory in MB per cpu allocated


SOURCE_DIR=${SOURCE_DIR:-"/home/maverkiou/zavou/style_detection/minkoski_pytorch/vae"}
PY_EXE=${PY_EXE:-"/home/maverkiou/miniconda2/envs/py3-mink/bin/python"}

DATA_DIR=${DATA_DIR:-"/mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components"}
DATASET=${DATASET:-"ComponentObjDataset"}
TRAIN_SPLIT_FILE=${TRAIN_SPLIT_FILE:-"/mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/splits/buildnet_component_max_train.txt"}
MAX_ITER=${MAX_ITER:-30000}
VAL_FREQ=${VAL_FREQ:-1000}
SAVE_FREQ=${SAVE_FREQ:-1000}
STAT_FREQ=${STAT_FREQ:-100}
LR=${LR:-1e-2}
LOG_DIR=${LOG_DIR:-/mnt/nfs/work1/kalo/maverkiou/zavou/mink_results/vae}

export PYTHONUNBUFFERED="True"
export PYTHONPATH="${PYTHONPATH}:${SOURCE_DIR}"

args="--train --lr ${LR} --max_visualization 10"
args="$args --data_dir ${DATA_DIR}"
args="$args --dataset ${DATASET}"
args="$args --train_split_file ${TRAIN_SPLIT_FILE}"
args="$args --max_iter ${MAX_ITER}"
args="$args --val_freq ${VAL_FREQ}"
args="$args --save_freq ${SAVE_FREQ}"
args="$args --stat_freq ${STAT_FREQ}"
args="$args --log_dir ${LOG_DIR}"


echo "cd ${SOURCE_DIR} && ${PY_EXE} vae.py $args"
cd ${SOURCE_DIR} && ${PY_EXE} vae.py $args 2>&1

