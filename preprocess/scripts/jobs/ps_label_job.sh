#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=longq
#SBATCH --time=7-00:00


JOB_NAME=$1
PY_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDING_FILE=$4
DATA_ROOT_DIR=$5
REPO=$6
INIT_SAMPLES=$7
PLY_DIR_PREFIX=$8
NUM_PROCESSES=$9
SAMPLE_POINTS_DIR=${10}
OBJ_DIR=${11}


args="--num_processes ${NUM_PROCESSES}"
args="$args --buildings_csv ${BUILDING_FILE}"
args="$args --obj_dir ${OBJ_DIR}"
args="$args --ply_dir ${SAMPLE_POINTS_DIR}/${PLY_DIR_PREFIX}_${INIT_SAMPLES}"
args="$args --pts_dir ${SAMPLE_POINTS_DIR}/point_cloud_${INIT_SAMPLES}"
args="$args --out_dir ${SAMPLE_POINTS_DIR}/normal_color_semlabel_${INIT_SAMPLES}"


echo "Calling cd ${STYLE_REPO}/preprocess/point_cloud_generation && ${PY_EXE} add_semantic_label.py $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/point_cloud_generation && ${PY_EXE} add_semantic_label.py $args
