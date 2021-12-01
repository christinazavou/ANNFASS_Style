GYPSUM_SOURCE_DIR ?= /home/maverkiou/zavou/decor-gan-private
GYPSUM_PY_EXE ?= /home/maverkiou/miniconda2/envs/decorgan/bin/python
GYPSUM_BLENDER_EXE ?= /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender
GYPSUM_DATA_ROOT ?= /mnt/nfs/work1/kalo/maverkiou/zavou/data
GYPSUM_LOGS_DIR ?= /mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs
TURING_SOURCE_DIR ?= /home/czavou01/decor-gan-private
TURING_PY_EXE ?= /home/czavou01/miniconda3/envs/decorgan/bin/python
TURING_LOGS_DIR ?= /home/czavou01/decorgan-logs
TURING_MODELS_DIR ?= /home/czavou01/decorgan-models
TURING_DATA_ROOT ?= /home/czavou01/ANNFASS_DATA
PARTITION ?= titanx-long
SPECIFIC_NODE_ARG ?=""
LOCAL_SOURCE_DIR ?= /media/graphicslab/BigData/zavou/decor-gan-private
LOCAL_PY_EXE ?= /home/graphicslab/miniconda3/envs/decorgan/bin/python
LOCAL_BLENDER_EXE ?= /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender
LOCAL_DATA_ROOT ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA
LOCAL_LOGS_DIR ?= /media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs
LOCAL_MODELS_DIR ?= /media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/local
OUT_DIR ?= chair_model
REPO ?= BUILDNET_Buildings
UNIQUE_DIR ?= groups_june17_unique_point_clouds
UNIFIED_NORMALIZED_DIR ?= groups_june17_uni_nor_components
TOTAL_JOBS ?= 6
ONLY_UNIQUE ?= True
BUILDING_CSV ?= "none"
#BUILDING_CSV ?= "/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style.csv"
# ----------------------------------------------------------------------------------------------------------------
build:
	cd ../data_preparation && python setup.py build_ext --inplace && chmod 777 binvox
	cd .. && python setup.py build_ext --inplace
# ----------------------------------------------------------------------------------------------------------------
SOURCE_DIR ?= $(GYPSUM_SOURCE_DIR)
PY_EXE ?= $(GYPSUM_PY_EXE)
DATA_ROOT ?= $(GYPSUM_DATA_ROOT)
LOGS_DIR ?= $(GYPSUM_LOGS_DIR)
DISPLAY ?= 50
TARGET_DIR ?= $(LOGS_DIR)/preprocessed_data

preprocess-some-building-components:
	mkdir -p "$(TARGET_DIR)" \
		&& export SOURCE_DIR=$(SOURCE_DIR) \
		&& export PY_EXE="$(PY_EXE)" \
		&& export SOURCE_ROOT="$(DATA_ROOT)/$(REPO)" \
		&& export TARGET_DIR="$(TARGET_DIR)" \
		&& export UNIQUE_DIR="$(DATA_ROOT)/$(REPO)/$(UNIQUE_DIR)" \
		&& export ONLY_UNIQUE="$(ONLY_UNIQUE)" \
		&& export BUILDING_CSV="$(BUILDING_CSV)" \
		&& export CLASS_ID=$(UNIFIED_NORMALIZED_DIR) \
		&& export PARTITION=$(PARTITION) \
		&& export TOTAL_JOBS=$(TOTAL_JOBS) \
		&& export DISPLAY=$(DISPLAY) \
		&& sh ./parallel_preprocess_some_components.sh

replace-correct:
	cd ../data_preparation \
		&& $(PY_EXE) replace_correct_components.py \
			--correct_dir "$(LOGS_DIR)/preprocessed_data/$(UNIFIED_NORMALIZED_DIR)_volume" \
			--wrong_dir "$(LOGS_DIR)/preprocessed_data/$(UNIFIED_NORMALIZED_DIR)"

buildnet-components-splits-gypsum:
	cd ../data_preparation \
		&& $(GYPSUM_PY_EXE) create_txt_files.py \
			"$(GYPSUM_LOGS_DIR)/preprocessed_data/groups_june17_uni_nor_components" \
			"$(GYPSUM_DATA_ROOT)/BUILDNET_Buildings/buildings_religious.csv" \
			"$(GYPSUM_LOGS_DIR)/splits/buildnet_groups_june17_uni_nor_components_max.txt" \
			"$(GYPSUM_DATA_ROOT)/BUILDNET_Buildings/groups_june17_unique_point_clouds"
buildnet-components-splits-local:
	cd ../data_preparation \
		&& $(LOCAL_PY_EXE) create_txt_files.py \
			"$(LOCAL_LOGS_DIR)/preprocessed_data/groups_june17_uni_nor_components" \
			"$(LOCAL_DATA_ROOT)/BUILDNET_Buildings/buildings_religious.csv" \
			"$(LOCAL_LOGS_DIR)/splits/buildnet_groups_june17_uni_nor_components_max.txt" \
			"$(LOCAL_DATA_ROOT)/BUILDNET_Buildings/groups_june17_unique_point_clouds"
# ----------------------------------------------------------------------------------------------------------------

MODELS_DIR ?= /mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/experimental_models
LOGS_DIR ?= /mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs
TAU ?= 0.1
NT_XENT_FACTOR ?= 1

LOGS_DIR := $(TURING_LOGS_DIR)
MODELS_DIR := $(TURING_MODELS_DIR)
CONFIG_FILES := "/home/czavou01/decor-gan-private/settings/turing1/adj_patchD3_16x10.yml,/home/czavou01/decor-gan-private/settings/turing1/adj_patchD3_32x8.yml,/home/czavou01/decor-gan-private/settings/turing1/adj_patchD3_64x6.yml,/home/czavou01/decor-gan-private/settings/turing1/adj_patchD3_128x2.yml"
GPU := 0
JOB_NAME := decorgan
YML ?= finetune/chair/adain_p2_in16_out128_g32d32.yml

run-config-gypsum:
	export SOURCE_DIR=$(GYPSUM_SOURCE_DIR) \
		&& export PY_EXE=$(GYPSUM_PY_EXE) \
		&& export CONFIG_YML="/home/maverkiou/zavou/decor-gan-private/settings/gypsum/$(YML)" \
		&& sbatch -J $(JOB_NAME) run_one_config_gypsum.sh

#CONFIG ?= finetune/chair/adain_p2_in16_out128_g32d32
#MAIN_FILE ?= mymain
CONFIG ?= cache_data/shapenet_chair/i16o128/rot
MAIN_FILE ?= dataset_rot
run-on-turing:
	mkdir -p $(CONFIG) && cd $(CONFIG) \
		&& export SOURCE_DIR=$(TURING_SOURCE_DIR) \
		&& export PY_EXE=$(TURING_PY_EXE) \
		&& export CONFIGS_DIR="/home/czavou01/decor-gan-private/settings/turing1" \
		&& export CONFIG=$(CONFIG) \
		&& export GPU=$(GPU) \
		&& export MAIN_FILE=$(MAIN_FILE) \
		&& sbatch -J $(JOB_NAME) $(SPECIFIC_NODE_ARG) $(TURING_SOURCE_DIR)/scripts/run_given_config_turing.sh

