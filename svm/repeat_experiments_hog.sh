#!/bin/bash

MODEL=hog
SVM_IMPL=hog_unique
OVERRIDE_LABELS=False
OVERRIDE_SVM=False
MODE=images

FEATURES_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_renderings/materials_on_daylight,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/groups_renderings/materials_on_daylight
COMPONENTS_CSV=/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv


IGNORE_CLASSES=Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown
UNIQUE_DIRS_ARGS="--unique_dirs /media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds"

make csv-generation FEATURES_DIR=${FEATURES_DIR} MODEL=${MODEL} OVERRIDE_LABELS=${OVERRIDE_LABELS} COMPONENTS_CSV=${COMPONENTS_CSV} EXPERIMENT_REPEAT=0 MODE=${MODE}

make run-svm-wrapper MODEL=${MODEL} SVM_IMPL=${SVM_IMPL} IGNORE_CLASSES=${IGNORE_CLASSES} UNIQUE_DIRS_ARGS="${UNIQUE_DIRS_ARGS}" EXPERIMENT_REPEAT=0 OVERRIDE_SVM=${OVERRIDE_SVM}

make csv-generation FEATURES_DIR=${FEATURES_DIR} MODEL=${MODEL} OVERRIDE_LABELS=${OVERRIDE_LABELS} COMPONENTS_CSV=${COMPONENTS_CSV} EXPERIMENT_REPEAT=1 MODE=${MODE}

make run-svm-wrapper MODEL=${MODEL} SVM_IMPL=${SVM_IMPL} IGNORE_CLASSES=${IGNORE_CLASSES} UNIQUE_DIRS_ARGS="${UNIQUE_DIRS_ARGS}" EXPERIMENT_REPEAT=1 OVERRIDE_SVM=${OVERRIDE_SVM}

make csv-generation FEATURES_DIR=${FEATURES_DIR} MODEL=${MODEL} OVERRIDE_LABELS=${OVERRIDE_LABELS} COMPONENTS_CSV=${COMPONENTS_CSV} EXPERIMENT_REPEAT=2 MODE=${MODE}

make run-svm-wrapper MODEL=${MODEL} SVM_IMPL=${SVM_IMPL} IGNORE_CLASSES=${IGNORE_CLASSES} UNIQUE_DIRS_ARGS="${UNIQUE_DIRS_ARGS}" EXPERIMENT_REPEAT=2 OVERRIDE_SVM=${OVERRIDE_SVM}

make csv-generation FEATURES_DIR=${FEATURES_DIR} MODEL=${MODEL} OVERRIDE_LABELS=${OVERRIDE_LABELS} COMPONENTS_CSV=${COMPONENTS_CSV} EXPERIMENT_REPEAT=3 MODE=${MODE}

make run-svm-wrapper MODEL=${MODEL} SVM_IMPL=${SVM_IMPL} IGNORE_CLASSES=${IGNORE_CLASSES} UNIQUE_DIRS_ARGS="${UNIQUE_DIRS_ARGS}" EXPERIMENT_REPEAT=3 OVERRIDE_SVM=${OVERRIDE_SVM}

make csv-generation FEATURES_DIR=${FEATURES_DIR} MODEL=${MODEL} OVERRIDE_LABELS=${OVERRIDE_LABELS} COMPONENTS_CSV=${COMPONENTS_CSV} EXPERIMENT_REPEAT=4 MODE=${MODE}

make run-svm-wrapper MODEL=${MODEL} SVM_IMPL=${SVM_IMPL} IGNORE_CLASSES=${IGNORE_CLASSES} UNIQUE_DIRS_ARGS="${UNIQUE_DIRS_ARGS}" EXPERIMENT_REPEAT=4 OVERRIDE_SVM=${OVERRIDE_SVM}
