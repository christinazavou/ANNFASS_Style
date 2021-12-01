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


args="-buildings_csv ${BUILDINGS_CSV}"
args="$args -root_data ${ROOT_DATA}"
args="$args -obj_dir_in ${OBJ_DIR}"
args="$args -obj_dir_out ${OBJ_DIR}Rotated"


echo "Calling cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python rotate_buildings.py -- $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python rotate_buildings.py -- $args
