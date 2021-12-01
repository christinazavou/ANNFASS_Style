#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


JOB_NAME=$1
PY_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
GDREPO=$5
ROOT_DATA=$6
GROUPS_DIR=$7

args="--buildings_csv ${ROOT_DATA}/${BUILDINGS_CSV}"
args="$args --root_dir ${ROOT_DATA}"
args="$args --gdrepo ${GDREPO}"
args="$args --groups_path ${GROUPS_DIR}"
args="$args --logs_dir ${ROOT_DATA}/upload_groups_logs"

echo "Calling cd ${STYLE_REPO}/preprocess/google_drive && ${PY_EXE} upload_groups.py $args" > ${LOG_FILE}
cd ${STYLE_REPO}/preprocess/google_drive && ${PY_EXE} upload_groups.py $args
