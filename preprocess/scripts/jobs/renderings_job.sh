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
USE_GPU_CYCLES=$8
RENDERS_DIR=$9
VIEWPOINTS_DIR=${10}
UNIQUE_DIR=${11}
PLY_DIR=${12}
RENDER_MODE=${13}


args="-BUILDINGS_CSV ${BUILDINGS_CSV}"
args="$args -OBJ_DIR ${OBJ_DIR}"
args="$args -GROUPS_DIR ${GROUP_DIR}"
args="$args -LOGS_DIR ${ROOT_DATA}/logs_renderings"
args="$args -MODE 2"
args="$args -RENDER_MODE ${RENDER_MODE}"
args="$args -RENDERS_DIR ${RENDERS_DIR}"
args="$args -VIEWPOINTS_DIR ${VIEWPOINTS_DIR}"
args="$args -UNIQUE_DIR ${UNIQUE_DIR}"
args="$args -PLY_DIR ${PLY_DIR}"
args="$args -USE_GPU_CYCLES ${USE_GPU_CYCLES}"

echo "Calling cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python viewpoints_and_renderings.py -- $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python viewpoints_and_renderings.py -- $args
