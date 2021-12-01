#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


JOB_NAME=$1
PY_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
ROOT_DIR=$5
PLY_PER_COMPONENT_DIR=$6
OUT_DIR=$7
DEBUG=$8


args="--root_dir ${ROOT_DIR}"
args="$args --buildings_csv ${BUILDINGS_CSV}"
#args="$args --obj_dir ${OBJ_DIR}"
#args="$args --points_dir ${POINTS_DIR}"
#args="$args --faces_dir ${FACES_DIR}"
#args="$args --groups_dir ${GROUPS_DIR}"
args="$args --ply_per_component_dir ${PLY_PER_COMPONENT_DIR}"
args="$args --out_dir ${OUT_DIR}"
args="$args --debug ${DEBUG}"


echo "Calling ${STYLE_REPO}/preprocess && ${PY_EXE} unique_components.py $args" > $LOG_FILE
cd ${STYLE_REPO}/preprocess && ${PY_EXE} unique_components.py $args
