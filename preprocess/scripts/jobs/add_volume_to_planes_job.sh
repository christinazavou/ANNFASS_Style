#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


JOB_NAME=$1
BLENDER_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
OBJ_DIR_IN=$5
ROOT_DATA=$6
OBJ_DIR_OUT=$7


args="-buildings_csv ${BUILDINGS_CSV}"
args="$args -root_data ${ROOT_DATA}"
args="$args -in_dir ${OBJ_DIR_IN}"
args="$args -out_dir ${OBJ_DIR_OUT}"
args="$args -logs_dir ${ROOT_DATA}/logs_volume"


echo "Calling cd ${STYLE_REPO}/preprocess/blender/buildnet && ${BLENDER_EXE} -b -noaudio --python add_volume_to_planes.py -- $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/blender/buildnet && ${BLENDER_EXE} -b -noaudio --python add_volume_to_planes.py -- $args
