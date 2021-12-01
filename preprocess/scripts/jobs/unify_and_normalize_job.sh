#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


JOB_NAME=$1
BLENDER_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
OBJ_DIR=$5
ROOT_DATA=$6
GROUP_DIR=$7
OUT_DIR=$8

args="-buildings_csv ${BUILDINGS_CSV}"
args="$args -root_data ${ROOT_DATA}"
args="$args -obj_dir ${OBJ_DIR}"
args="$args -group_dir ${GROUP_DIR}"
args="$args -logs_dir ${ROOT_DATA}/logs_unifynormalize"
args="$args -out_dir ${ROOT_DATA}/${OUT_DIR}"
args="$args -unify True"
args="$args -normalize True"


echo "Calling cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python unify_and_normalize_components.py -- $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python unify_and_normalize_components.py -- $args
