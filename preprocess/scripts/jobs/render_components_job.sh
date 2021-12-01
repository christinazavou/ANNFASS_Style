#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


JOB_NAME=$1
BLENDER_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
ROOT_DATA=$5
COMPONENTS_DIR=$6
UNIQUE_DIR=$7


args="-BUILDINGS_CSV ${ROOT_DATA}/${BUILDINGS_CSV}"
args="$args -ROOT_DIR ${ROOT_DATA}"
args="$args -COMPONENTS_DIR ${COMPONENTS_DIR}"
args="$args -UNIQUE_DIR ${UNIQUE_DIR}"
args="$args -LOGS_DIR ${ROOT_DATA}/logs_june17"
args="$args -RENDER_MODE 1"


echo "Calling cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python render_components.py -- $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/blender && ${BLENDER_EXE} -b -noaudio --python render_components.py -- $args
