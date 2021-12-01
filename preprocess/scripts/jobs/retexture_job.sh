#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


JOB_NAME=$1
BLENDER_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
OBJ_DIR=$5


args="-buildings_csv ${BUILDINGS_CSV}"
args="$args -obj_dir_in ${OBJ_DIR}"

echo "Calling cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python retexture_obj.py -- $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python retexture_obj.py -- $args
