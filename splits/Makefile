SHELL := /bin/bash

CONDA_ENV ?= py3-mink
CONDA_PATH ?= /home/graphicslab/miniconda3
ROOT_DATA ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA
LOGS_DIR ?= /media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs

REPO ?= ANNFASS_Buildings_may
SPLIT_DIR ?= combined_final_splits
CONTENT_DIR ?= samplePoints/stylePly_cut10.0K_pgc_content512
STYLE_DIR ?= samplePoints/stylePly_cut10.0K_pgc_style4096
PLY_DIR ?= samplePoints_refinedTextures/normal_color_semlabel_1000K
UNIQUE_DIR ?= unique_point_clouds
BUILDING_FILE ?= buildings.csv
PLY_DIRS ?= $(REPO)/$(PLY_DIR)
UNIQUE_DIRS ?= "$(REPO)/$(UNIQUE_DIR),"
NUM_FOLDS ?= 1

PY_EXE := $(CONDA_PATH)/envs/$(CONDA_ENV)/bin/python
STYLE_STATS_FILE ?= Combined_Buildings/stats/style_stats_selected_parts_31102021_building.csv
PARTS_ARGS ?= ""
ONLY_UNIQUE ?= True
TRAIN_PCT ?= 0.8


classification_cross_val_splits:
	export PATH=$(CONDA_PATH)/bin/:$$PATH \
		&& $(PY_EXE) classification_cross_val_splits.py \
			--style_stats_file $(ROOT_DATA)/$(STYLE_STATS_FILE) \
			--out_dir $(LOGS_DIR)/$(SPLIT_DIR)


ply_split_files:
	export PATH=$(CONDA_PATH)/bin/:$$PATH \
		&& $(PY_EXE) ply_splits_generation.py \
			--root_dir $(ROOT_DATA) \
			--ply_dirs $(PLY_DIRS) \
			--unique_dirs $(UNIQUE_DIRS) \
			--split_root $(LOGS_DIR)/$(SPLIT_DIR) \
			--splits_json $(LOGS_DIR)/$(SPLIT_DIR)/classification_cross_val.json \
			--num_folds $(NUM_FOLDS)


buildnet_content_style_splits:
	$(PY_EXE) buildnet_content_style_splits.py \
		--root_dir $(ROOT_DATA) \
		--logs_dir $(LOGS_DIR) \
		--repo $(REPO) \
		--content_dir $(CONTENT_DIR) \
		--style_dir $(STYLE_DIR) \
	  	--splits_dir $(SPLIT_DIR)_content_style \
	  	--unique_dir $(UNIQUE_DIR) \
	  	--buildings_csv $(BUILDING_FILE) \
	  	--only_unique $(ONLY_UNIQUE) \
	  	--train_pct $(TRAIN_PCT) $(PARTS_ARGS)


annfass_content_style_splits:
	$(PY_EXE) annfass_content_style_splits.py \
		--root_dir $(ROOT_DATA) \
		--logs_dir $(LOGS_DIR) \
		--repo $(REPO) \
		--content_dir $(CONTENT_DIR) \
		--style_dir $(STYLE_DIR) \
	  	--splits_dir $(SPLIT_DIR)_content_style \
	  	--splits_json $(SPLIT_DIR)/classification_cross_val.json

