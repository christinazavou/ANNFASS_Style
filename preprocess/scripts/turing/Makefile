SHELL := /usr/bin/bash # otherwise we can't use "source"

CONDA_ENV ?= style_detect_env
ROOT_DIR := /mnt/beegfs/czavou01/ANNFASS_DATA
CONDA_PATH := /home/czavou01/miniconda3
PY_EXE := ${CONDA_PATH}/envs/$(CONDA_ENV)/bin/python
STYLE_REPO := /home/czavou01/style_detection
SFE_EXE := ../../mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter
BLENDER_EXE ?= /home/czavou01/blender-2.91.0-linux64/blender

REPO ?= BUILDNET_Buildings
SAMPLES ?= 1000000
INIT_SAMPLES ?= 1000K
OBJ_DIR ?= $(ROOT_DIR)/$(REPO)/normalizedObj
GROUPS_DIR ?= $(ROOT_DIR)/$(REPO)/groups_june17
UNIQUE_DIR ?= $(ROOT_DIR)/$(REPO)/groups_june17_unique_point_clouds
VIEWPOINTS_DIR ?= $(ROOT_DIR)/$(REPO)/groups_june17_viewpoints
RENDERINGS_DIR ?= $(ROOT_DIR)/$(REPO)/groups_june17_renderings
PLY_N_DIR ?= $(ROOT_DIR)/$(REPO)/normalizedPly
SAMPLES_OUT_DIR ?= $(ROOT_DIR)/$(REPO)/samplePoints_dokimi
PTS_ON_STYLE ?= False
ON_GPU ?= True
NUM_PROCESSES ?= 8
START_IDX ?= 0
END_IDX ?= 30
BUILDING_FILE ?= $(ROOT_DIR)/$(REPO)/buildings_to_sfe
STEP_SIZE ?= 400
MEMORY ?= 28GB
GPU ?= 0
NODE ?= gpu-0-1

point-clouds:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="SFE_$(REPO)" \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/mesh_sampling_job.sh" \
		&& export USE_FUNCTION=run_meshsampling \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--mem $(MEMORY) --cpus-per-task=$(NUM_PROCESSES) --partition=COMPUTE" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export OBJ_DIR=$(OBJ_DIR)_refinedTexturesRotated \
		&& export REMOVE=False \
		&& export OVERRIDE=False \
		&& export ON_STYLE=$(PTS_ON_STYLE) \
		&& export NUM_PROCESSES=$(NUM_PROCESSES) \
		&& export SAMPLE_POINTS_DIR=$(SAMPLES_OUT_DIR) \
		&& bash ../../schedule_same_jobs.sh

retexture:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="RETEXTURE_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/retexture_job.sh" \
		&& export USE_FUNCTION=run_retexture \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--mem-per-cpu=100000 --partition GPU -w $(NODE)" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export GPU=$(GPU) \
		&& bash ../../schedule_same_jobs.sh
# NOTE: after retexture on cluster and bringing them locally we need to run the change_mtl_locations.py !!

buildnet-viewpoints:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="BVIEW_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/viewpoints_job.sh" \
		&& export USE_FUNCTION=run_viewpoints \
		&& export BLENDER_EXE=/home/czavou01/blender-2.93.5-linux-x64/blender \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--mem-per-cpu=100000 --ntasks=8 --partition GPU -w $(NODE)" \
		&& export GPU=$(GPU) \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export UNIQUE_DIR=$(UNIQUE_DIR) \
		&& export VIEWPOINTS_DIR=$(VIEWPOINTS_DIR) \
		&& export RENDERINGS_DIR=$(RENDERINGS_DIR) \
		&& export PLY_N_DIR=$(PLY_N_DIR) \
		&& export ON_GPU=$(ON_GPU) \
		&& bash ../../schedule_same_jobs.sh