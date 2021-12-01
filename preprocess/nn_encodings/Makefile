SHELL := /bin/bash

CONDA_ENV := style_detect_env
CONDA_PATH:=/home/graphicslab/miniconda3

ROOT_DIR ?= /media/graphicslab/BigData1/zavou/ANNFASS_DATA
LOCAL_STYLE_DIR ?= /media/graphicslab/BigData1/zavou/style_detection

LOGS_DIR ?= $(LOCAL_STYLE_DIR)/logs

REPO ?= ANNFASS_Buildings_march
OBJ_DIR ?= normalizedObj
GROUPS_DIR ?= groups
SAMPLES_DIR ?= samplePoints
FACES_DIR ?= faces_10000K

BUILDING_FILE ?= $(ROOT_DIR)/$(REPO)/buildings.csv
FEATURES_DIR ?= buildnet_minkowski_ps/buildnet_ply100Kmarios/StylenetVoxelization0_01Dataset/PS-Res16UNet34A-MultiGpu/b32-i120000/latent_features_annfass/test_split

encodings_per_component:
	source $(CONDA_PATH)/etc/profile.d/conda.sh \
	&& conda activate $(CONDA_ENV) \
	&& python main.py \
		--root_dir $(ROOT_DIR) \
		--features_dir $(FEATURES_DIR) \
	  	--face_indices_dir $(REPO)/$(SAMPLES_DIR)/$(FACES_DIR) \
		--normalizedObj_dir $(REPO)/$(OBJ_DIR) \
    	--groups_dir $(REPO)/$(GROUPS_DIR) \
  		--buildings_csv $(BUILDING_FILE)

# todo (july): add rnv in main.py and print also points per component in stats file, and then remove main_no_groups.py
#RIDGE_VALLEY_DIR ?= ridge_or_valley_10000K
#EXPORTED_FEATURES ?= layer_n-2_features
#encodings_per_component_no_groups:
#	source ${CONDA_PATH}/etc/profile.d/conda.sh \
#	&& conda activate style_detect_env \
#	&& python main_no_groups.py \
#		--root_dir ${ROOT_DIR} \
#		--features_dir ${EXPORTED_FEATURES_DIR} \
#	  	--face_indices_dir ${REPO}/samplePoints/${FACES_DIR} \
#	  	--rnv_dir ${REPO}/samplePoints/${RIDGE_VALLEY_DIR} \
#		--normalizedObj_dir ${REPO}/normalizedObj \
#  		--buildings_csv ${ROOT_DIR}/${REPO}/buildings.csv

clean:
	rm -rf ${OUTPUT_DIR}