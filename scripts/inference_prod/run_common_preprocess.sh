#!/bin/bash


STYLE_DETECTION_REPO=$1  # /style_workdir/style_detection
BLENDER_EXE=$2  # /style_workdir/software/blender-2.91.0-linux64/blender
CONDA_PATH=$3  # /style_workdir/software/miniconda3

make create-input-file STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO}
make triangulate STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO} BLENDER_EXE=${BLENDER_EXE}
make normalize STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO} BLENDER_EXE=${BLENDER_EXE} CONDA_PATH=${CONDA_PATH}
make retexture STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO} BLENDER_EXE=${BLENDER_EXE}
make point-cloud STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO} CONDA_PATH=${CONDA_PATH}
make downsample-pts STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO} CONDA_PATH=${CONDA_PATH}
make color-point-cloud STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO} CONDA_PATH=${CONDA_PATH}

