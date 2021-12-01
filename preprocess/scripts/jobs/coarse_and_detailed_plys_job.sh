#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


JOB_NAME=$1
PY_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
ROOT_DATA=$5
REPO=$6
PLY_PER_COMPONENT_DIR=$7
DETAIL_SAMPLES=$8
COARSE_SAMPLES=$9


args="--root ${ROOT_DATA}"
args="$args --repo ${REPO}"
args="$args --ply_per_component_dir ${PLY_PER_COMPONENT_DIR}"
args="$args --buildings_csv ${BUILDINGS_CSV}"
args="$args --detail_samples ${DETAIL_SAMPLES}"
args="$args --coarse_samples ${COARSE_SAMPLES}"


echo "Calling cd ${STYLE_REPO}/preprocess/point_cloud_generation && ${PY_EXE} coarse_and_detailed_plys.py $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/point_cloud_generation && ${PY_EXE} coarse_and_detailed_plys.py $args
