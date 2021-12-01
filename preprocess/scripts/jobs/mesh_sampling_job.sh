#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


JOB_NAME=$1
PY_EXE=$2
STYLE_REPO=$3
LOG_FILE=${JOB_NAME}.log


BUILDINGS_CSV=$4
OBJ_DIR=$5
NUM_PROCESSES=$6
NUM_SAMPLES=$7
EXPORT_CURVATURES=$8
EXPORT_PCA=$9
OVERRIDE=${10}
REMOVE=${11}
ON_STYLE=${12}
OUT_SAMPLES_DIR=${13}


DEBUG=False
SFE=${STYLE_REPO}/preprocess/mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter


args="--num_processes ${NUM_PROCESSES}"
args="$args --num_samples ${NUM_SAMPLES}"
args="$args --buildings_csv ${BUILDINGS_CSV}"
args="$args --debug ${DEBUG}"
args="$args --sfe ${SFE}"
args="$args --obj_dir ${OBJ_DIR}"
args="$args --export_curvatures ${EXPORT_CURVATURES}"
args="$args --export_pca ${EXPORT_PCA}"
args="$args --override ${OVERRIDE}"
args="$args --remove ${REMOVE}"
args="$args --stylistic_selection ${ON_STYLE}"
args="$args --out_samples_dir ${OUT_SAMPLES_DIR}"

echo "Calling ${STYLE_REPO}/preprocess/point_cloud_generation && ${PY_EXE} create_point_clouds.py $args" > $LOG_FILE
cd ${STYLE_REPO}/preprocess/point_cloud_generation && ${PY_EXE} create_point_clouds.py $args
