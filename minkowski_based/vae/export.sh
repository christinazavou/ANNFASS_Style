#!/bin/bash


SOURCE_DIR=${SOURCE_DIR:-"/home/maverkiou/zavou/style_detection/minkoski_pytorch/vae"}
PY_EXE=${PY_EXE:-"/home/maverkiou/miniconda2/envs/py3-mink/bin/python"}

DATA_DIR=${DATA_DIR:-"/mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components"}
DATASET=${DATASET:-"ComponentObjDataset"}
VAL_SPLIT_FILE=${VAL_SPLIT_FILE:-"/mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/splits/buildnet_component_max_train.txt"}
LOG_DIR=${LOG_DIR:-/mnt/nfs/work1/kalo/maverkiou/zavou/mink_results/vae}
WEIGHTS_CKPT=${WEIGHTS_CKPT:-model_vae.pth}
ENCODINGS_DIR=${ENCODINGS_DIR:-encodings}

export PYTHONUNBUFFERED="True"
export PYTHONPATH="${PYTHONPATH}:${SOURCE_DIR}"

args="--export"
args="$args --data_dir ${DATA_DIR}"
args="$args --dataset ${DATASET}"
args="$args --val_split_file ${VAL_SPLIT_FILE}"
args="$args --log_dir ${LOG_DIR}"
args="$args --weights ${WEIGHTS_CKPT}"
args="$args --encodings_dir ${ENCODINGS_DIR}"


echo "cd ${SOURCE_DIR} && ${PY_EXE} vae.py $args"
cd ${SOURCE_DIR} && ${PY_EXE} vae.py $args 2>&1

