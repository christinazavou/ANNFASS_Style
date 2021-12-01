#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=longq
#SBATCH --time=7-00:00

JOB_NAME=$1
PY_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDING_FILE=$4
INIT_SAMPLES=$5
PLY_DIR_PREFIX=$6
GROUPS_DIR=$7
WITH_GROUPS=${8}
CUT_AT=${9}
RIDGE_VALLEY_DIR=${10}
NUM_PROCESSES=${11}
PER_COMPONENT=${12}
SAMPLE_POINTS_DIR=${13}
OBJ_DIR=${14}
COLOR_DIR=${15}


args="--obj_dir ${OBJ_DIR}"
args="$args --pts_dir ${SAMPLE_POINTS_DIR}/point_cloud_${INIT_SAMPLES}"
args="$args --ply_dir_prefix ${PLY_DIR_PREFIX}"
args="$args --buildings_csv ${BUILDING_FILE}"
args="$args --cut_at ${CUT_AT}"
args="$args --rnv ${RIDGE_VALLEY_DIR}"
args="$args --color ${COLOR_DIR}"
args="$args --num_processes ${NUM_PROCESSES}"
args="$args --per_component ${PER_COMPONENT}"


if [ "${WITH_GROUPS}" == "True" ];then
  args="$args --groups_dir ${GROUPS_DIR}"
fi


echo "Calling cd ${STYLE_REPO}/preprocess/point_cloud_generation && ${PY_EXE} pts2plys.py $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/point_cloud_generation && ${PY_EXE} pts2plys.py $args
