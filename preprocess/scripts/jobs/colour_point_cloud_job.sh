#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err

JOB_NAME=$1
PY_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
OBJ_DIR=$5
SAMPLES_OUT_DIR=$6
INIT_SAMPLES=$7


args="--objects_dir ${OBJ_DIR}"
args="$args --points_dir ${SAMPLES_OUT_DIR}/point_cloud_${INIT_SAMPLES}"
args="$args --faces_dir ${SAMPLES_OUT_DIR}/faces_${INIT_SAMPLES}"
args="$args --buildings_csv ${BUILDINGS_CSV}"
args="$args --output_dir ${SAMPLES_OUT_DIR}/colorPly_${INIT_SAMPLES}"

echo "Calling cd ${STYLE_REPO}/preprocess/colour_point_cloud && ${PY_EXE} main.py $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/colour_point_cloud && ${PY_EXE} main.py $args
