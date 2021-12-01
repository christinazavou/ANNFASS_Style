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
ON_GPU=$8


args="-buildings_csv ${BUILDINGS_CSV}"
args="$args -root_data ${ROOT_DATA}"
args="$args -obj_dir ${OBJ_DIR}"
args="$args -group_dir ${GROUP_DIR}"
args="$args -logs_dir ${ROOT_DATA}/buildnetgrouplogs"
args="$args -on_gpu ${ON_GPU}"
#args="$args -debug False"
args="$args -debug True"

echo "Calling cd ${STYLE_REPO}/preprocess/blender/buildnet && ${BLENDER_EXE} -b -noaudio --python group_as_component.py -- $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/blender/buildnet && ${BLENDER_EXE} -b -noaudio --python group_as_component.py -- $args
