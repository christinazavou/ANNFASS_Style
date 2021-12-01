#!/bin/bash


OCNN_REPO=$1  # /style_workdir/style_detection/ocnn_based
PY_EXE=$2  # /root/miniconda/envs/structure_ann/bin/python
STYLE_DETECTION_REPO=$3  #  /style_workdir/style_detection
BUILDING_NAME=$4  # 14_Stavros_tou_Misericou
BLENDER_EXE=$5

function encodings() {
  make create-input-file BUILDING_NAME=${BUILDING_NAME}
  make create-input-file-ocnn BUILDING_NAME=${BUILDING_NAME}
  make encode-with-ocnn-structure STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO} OCNN_REPO=${OCNN_REPO} PY_EXE=${PY_EXE}
}

function encodings_per_component(){
    make group STYLE_DETECTION_REPO=${STYLE_DETECTION_REPO} BLENDER_EXE=${BLENDER_EXE}
    make encodings-per-component
    make create-input-file-svm
}
