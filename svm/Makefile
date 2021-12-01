SHELL := /bin/bash

STYLE_DIR := /media/graphicslab/BigData1/zavou/style_detection
LOGS_DIR := /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final
ROOT_DATA := /media/graphicslab/BigData1/zavou/ANNFASS_DATA
CONDA_PATH := /home/graphicslab/miniconda3
CONDA_ENV := style_detect_env
PY_EXE:=$(CONDA_PATH)/envs/$(CONDA_ENV)/bin/python

COMPONENTS_CSV ?= $(ROOT_DATA)/Combined_Buildings/selected_components_with_style.csv

EXPERIMENT_REPEAT ?= 0
SPLITS_DIR=$(LOGS_DIR)/labeled_data/repeat_$(EXPERIMENT_REPEAT)

MODEL ?= 3daae_on_buildnet
FEATURES_DIR ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/3daae-results/buildnet/turing/aae/experiment_aae_buildnet1/encodingssetBC/epoch02000_z_e
PS_BASED ?= False
NUM_PROCESSES ?= 6
#SPECIFIC_DIR_ARGS ?= "--layer layer_n-2_features --point_reduce_method weighted_sum --component_reduce_method max"
SPECIFIC_DIR_ARGS ?=
UNIQUE_DIRS_ARGS ?=
IGNORE_CLASSES ?= "Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown"
SVM_IMPL ?= simple
OVERRIDE_LABELS ?= False
OVERRIDE_SVM ?= False
MODE ?= encodings

MODEL_AE_2D := freestyle/batch32epoch500data040221
MODEL_DIR_AE_2D := $(LOGS_DIR)/buildnet_ae/models/${MODEL_AE_2D}
ENC_NAME_AE_2D := encodings_on_data050221
ENCODINGS_AE_2D := $(LOGS_DIR)/annfass_svm/data_buildnet_ae/${MODEL_AE_2D}/${ENC_NAME_AE_2D}
MODELS_DIR_2D := $(LOGS_DIR)/annfass_svm/models/ae2d/${ENC_NAME_AE_2D}_from_${MODEL_AE_2D}

DATA_DIR_CURVATURES := $(LOGS_DIR)/${SPLITS_DIR}/${PLY_SAMPLES}
MODEL_DIR_CURVATURES := $(LOGS_DIR)/svm_curvatures/${PLY_SAMPLES}


#----------------------------------------------------------------------------------------------------------------------#
csv-generation:
	$(PY_EXE) split_csv_generation.py \
			--data_dirs $(FEATURES_DIR) \
			--out_dir $(SPLITS_DIR)_svm_from_$(MODEL)/data/as_is \
			--components_csv $(COMPONENTS_CSV) \
			--splits $(SPLITS_DIR)/classification_cross_val.json \
			--override_labels $(OVERRIDE_LABELS) \
			--mode $(MODE)

csv-generation-wrapper:
	export PATH=$(CONDA_PATH)/bin/:$$PATH \
		&& source activate $(CONDA_ENV) \
		&& python split_csv_generation_wrapper.py \
			--data_dir $(FEATURES_DIR) \
			--out_dir $(SPLITS_DIR)_svm_from_$(MODEL)/data \
			--py_exe $(PY_EXE) \
			--splits $(SPLITS_DIR)/classification_cross_val.json \
			--override_labels $(OVERRIDE_LABELS) \
			--components_csv $(COMPONENTS_CSV) \
			--part_segmentation_based $(PS_BASED) \
			$(SPECIFIC_DIR_ARGS)
#----------------------------------------------------------------------------------------------------------------------#
run-svm:
	export PYTHONPATH=$$PYTHONPATH:$(STYLE_DIR):$(STYLE_DIR)/svm \
		&& mkdir -p $(SPLITS_DIR)_svm_from_$(MODEL)/svm_$(SVM_IMPL) \
		&& $(PY_EXE) run_svm.py \
			--data_dir $(SPLITS_DIR)_svm_from_$(MODEL)/data/as_is \
			--models_dir $(SPLITS_DIR)_svm_from_$(MODEL)/svm_$(SVM_IMPL) \
			--ignore_classes $(IGNORE_CLASSES) \
			--avg_f1_nan 0 \
			--svm_impl $(SVM_IMPL) $(UNIQUE_DIRS_ARGS)\
			> $(SPLITS_DIR)_svm_from_$(MODEL)/svm_$(SVM_IMPL)/log.txt

run-svm-wrapper:
	export PYTHONPATH=$$PYTHONPATH:$(STYLE_DIR):$(STYLE_DIR)/svm \
		&& mkdir -p $(SPLITS_DIR)_svm_from_$(MODEL)/svm_$(SVM_IMPL) \
 		&& $(PY_EXE) run_svm_wrapper.py \
			--data_dir $(SPLITS_DIR)_svm_from_$(MODEL)/data \
			--models_dir $(SPLITS_DIR)_svm_from_$(MODEL)/svm_$(SVM_IMPL) \
			--py_exe $(PY_EXE) \
			--ignore_classes $(IGNORE_CLASSES) \
			--svm_impl $(SVM_IMPL) $(UNIQUE_DIRS_ARGS)\
			--num_processes $(NUM_PROCESSES) \
			--avg_f1_nan 0 \
			--override $(OVERRIDE_SVM) \
			> $(SPLITS_DIR)_svm_from_$(MODEL)/svm_$(SVM_IMPL)/log.txt
