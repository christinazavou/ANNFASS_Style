SHELL := /bin/bash # otherwise we can't use "source"

CONDA_PATH := /home/graphicslab/miniconda3
CONDA_ENV ?= style_detect_env
PY_EXE := ${CONDA_PATH}/envs/$(CONDA_ENV)/bin/python

SFE_EXE := ../../mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter
BLENDER_EXE ?= /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender
STYLE_DETECTION_REPO := /media/graphicslab/BigData1/zavou/style_detection
OCNN_REPO ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/O-CNN
OCNN_CHECKPOINTS ?= $(STYLE_DETECTION_REPO)/resources/trained_models/structure_ann/ocnn_HRNet_snapshots
OCNN_CHECKPOINT ?= $(OCNN_CHECKPOINTS)/no_colour_depth_6_maj_vote/best_total_loss.ckpt

BUILDING_NAME ?= 13_Kyrenia_Gate
DATA_DIR := $(STYLE_DETECTION_REPO)/scripts/inference_prod/sample_data
DATA_REPO := john_doe_buildings
OBJ_DIR ?= $(DATA_DIR)/$(DATA_REPO)/normalizedObj
BUILDING_FILE ?= $(DATA_DIR)/$(DATA_REPO)/inference_input
SAMPLES_OUT_DIR ?= $(DATA_DIR)/$(DATA_REPO)/samplePoints
INIT_SAMPLES ?= 1000K
OCNN_ENCODINGS_DIR ?= $(DATA_DIR)/$(DATA_REPO)/ocnn_encodings

# 0.01 SAMPLES <= generated points <= 1.01 SAMPLES ==> use SAMPLES:=generated points / 0.01
SAMPLES ?= 10000000
INIT_SAMPLES ?= 10000K
FINAL_SAMPLES ?= 100K
CUT_AT ?= 100000
PLY_DIR_PREFIX ?= ply
PER_COMPONENT ?= False
PTS_ON_STYLE ?= False
NUM_PROCESSES ?= 8
START_IDX ?= 0
END_IDX ?= 4
STEP_SIZE ?= 2
ON_GPU ?= False
WITH_GROUPS ?= True


create-input-file:
	echo "toinfer;$(BUILDING_NAME)" > $(DATA_DIR)/$(DATA_REPO)/inference_input.csv

# NOTE: triangulate will only support .fbx input files FIXME
triangulate:
	cd $(STYLE_DETECTION_REPO)/preprocess/scripts/local \
		&& make triangulate-annfass \
		STYLE_REPO=$(STYLE_DETECTION_REPO) \
		BLENDER_EXE=$(BLENDER_EXE) \
		ROOT_DIR=$(DATA_DIR) \
		ANNFASS_REPO=$(DATA_REPO) \
		BUILDING_FILE=$(BUILDING_FILE)

normalize:
	cd $(STYLE_DETECTION_REPO)/preprocess/scripts/local \
		&& make normalize-annfass \
		STYLE_REPO=$(STYLE_DETECTION_REPO) \
		PY_EXE=$(PY_EXE) \
		ROOT_DIR=$(DATA_DIR) \
		ANNFASS_REPO=$(DATA_REPO) \
		BUILDING_FILE=$(BUILDING_FILE)

retexture:
	cd $(STYLE_DETECTION_REPO)/preprocess/scripts/local \
		&& make retexture \
		STYLE_REPO=$(STYLE_DETECTION_REPO) \
		BLENDER_EXE=$(BLENDER_EXE) \
		ROOT_DIR=$(DATA_DIR) \
		REPO=$(DATA_REPO) \
		OBJ_DIR=$(OBJ_DIR) \
		START_IDX=0 \
		END_IDX=1 \
		BUILDING_FILE=$(BUILDING_FILE) >> $(DATA_DIR)/$(DATA_REPO)/logs/retexture.log

point-cloud:
	cd $(STYLE_DETECTION_REPO)/preprocess/scripts/local \
		&& make point-clouds \
		STYLE_REPO=$(STYLE_DETECTION_REPO) \
		PY_EXE=$(PY_EXE) \
		REPO=$(DATA_REPO) \
		OBJ_DIR=$(OBJ_DIR)_refinedTextures \
		BUILDING_FILE=$(BUILDING_FILE) \
		START_IDX=0 \
		END_IDX=1 \
		SAMPLES_OUT_DIR=$(SAMPLES_OUT_DIR)_refinedTextures

downsample-pts:
	cd $(STYLE_DETECTION_REPO)/preprocess/point_cloud_generation \
		&& $(PY_EXE) downsample_points.py \
		--input_cloud_dir=$(SAMPLES_OUT_DIR)_refinedTextures/point_cloud_$(INIT_SAMPLES) \
		--output_cloud_dir=$(SAMPLES_OUT_DIR)_refinedTextures/point_cloud_$(FINAL_SAMPLES) \
		--samples $(CUT_AT)  >> $(DATA_DIR)/$(DATA_REPO)/logs/downsample.log

color-point-cloud:
	cd $(STYLE_DETECTION_REPO)/preprocess/scripts/local \
		&& make colour-point-clouds \
		STYLE_REPO=$(STYLE_DETECTION_REPO) \
		PY_EXE=$(PY_EXE) \
		REPO=$(DATA_REPO) \
		OBJ_DIR=$(OBJ_DIR)_refinedTextures \
		BUILDING_FILE=$(BUILDING_FILE) \
		START_IDX=0 \
		END_IDX=1 \
		SAMPLES_OUT_DIR=$(SAMPLES_OUT_DIR)_refinedTextures \
		INIT_SAMPLES=$(FINAL_SAMPLES)

create-input-file-ocnn:
	echo "$(BUILDING_NAME).ply 50" > $(DATA_DIR)/$(DATA_REPO)/inference_input_ocnn.txt

# note: call this with CONDA_ENV tf1.14ocnnagain
encode-with-ocnn-structure:
	cd $(OCNN_REPO)/tensorflow/script \
		&& $(PY_EXE) run_seg_partnet_annfass.py \
		--config $(STYLE_DETECTION_REPO)/resources/trained_models/structure_ann/seg_hrnet_pts_inference.yaml \
		SOLVER.ckpt $(OCNN_CHECKPOINT) \
		DATA.test.location $(SAMPLES_OUT_DIR)/colorPly_100K \
		DATA.test.file_list $(DATA_DIR)/$(DATA_REPO)/inference_input_ocnn.txt \
		SOLVER.logdir $(OCNN_ENCODINGS_DIR) \
		SOLVER.test_iter 1
# NOTE: group will only support annfass-like data (assume well defined big components) FIXME
group:
	cd $(STYLE_DETECTION_REPO)/preprocess/scripts/local \
		&& make groups-annfass \
		STYLE_REPO=$(STYLE_DETECTION_REPO) \
		BLENDER_EXE=$(BLENDER_EXE) \
		ROOT_DIR=$(DATA_DIR) \
		ANNFASS_REPO=$(DATA_REPO) \
		OBJ_DIR=normalizedObj_refinedTextures_withStructure \
		GROUPS_DIR=groups_refinedTextures_withStructure \
		BUILDING_FILE=inference_input

encodings-per-component:
	cd $(STYLE_DETECTION_REPO)/preprocess/nn_encodings \
		&& make encodings_per_component \
		FEATURES_DIR=$(OCNN_ENCODINGS_DIR)/best_total_loss/encodings \
		ROOT_DIR=$(DATA_DIR) \
		REPO=$(DATA_REPO) \
		FACES_DIR=faces_100K \
		OBJ_DIR=normalizedObj_refinedTextures_withStructure \
		SAMPLES_DIR=samplePoints_refinedTextures \
		GROUPS_DIR=groups_refinedTextures_withStructure \
		BUILDING_FILE=$(BUILDING_FILE).csv

create-input-file-svm:
	cd $(STYLE_DETECTION_REPO)/svm \
		&& $(PY_EXE) csv_generation_inference.py \
		--data_dir $(OCNN_ENCODINGS_DIR)/best_total_loss/encodings/feature_concat/as_is_per_component \
		--out_file $(DATA_DIR)/$(DATA_REPO)/infer_style_per_component.csv \
		--elements column,dome,tower,window,door


