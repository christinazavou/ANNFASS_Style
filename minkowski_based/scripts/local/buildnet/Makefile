SHELL := /bin/bash

PY_EXE := /home/graphicslab/miniconda3/envs/py3-mink/bin/python
STYLE_ROOT := /media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection
LOGS_ROOT := $(STYLE_ROOT)/logs

DATA_NAME ?= buildnet
PLY_SAMPLES ?= ply_10K
SPLITS_SUFFIX ?= reconstruction_splits
SPLIT_TO_VAL ?= test
TRAIN_SPLIT ?= split_train_val_test
TEST_SPLIT ?= split_train_val_test

SPLITS_DIR := ${DATA_NAME}_${SPLITS_SUFFIX}

TEST_MODEL_DIR ?= b5-i500
TEST_DATA_NAME ?= annfass
TEST_SPLIT_SUFFIX ?= splits_march/ply_100K_cnscr/fold0
TEST_SPLIT_DIR := $(TEST_DATA_NAME)_$(TEST_SPLIT_SUFFIX)

AE_DIR := ${LOGS_ROOT}/${DATA_NAME}_minkowski_ae/${DATA_NAME}_${PLY_SAMPLES}
PS_DIR := $(LOGS_ROOT)/$(DATA_NAME)_minkowski_ps/$(DATA_NAME)_$(PLY_SAMPLES)
CLS_STYLE_DIR := $(LOGS_ROOT)/$(DATA_NAME)_minkowski_cls_style/$(DATA_NAME)_$(PLY_SAMPLES)

#AE_MODEL := Res16UNetAEL8B2
#AE_MODEL := HRNetAE3S2BD128
AE_MODEL ?= HRNetAE3S2BD256
#AE_MODEL := SmallNetAE
#AE_MODEL := HRNetAE1S2BD128
#AE_MODEL := HRNetAE3S3BND256
#AE_MODEL := HRNetAE3S2BND128
PS_MODEL ?= Res16UNet34A
#PS_MODEL ?= HRNet3S2BD256
CLS_RNV_MODEL ?= HRNet3S2BD256
CLS_MODEL ?= SmallNetStyleCls
BATCH_SIZE ?= 4
MAX_ITER ?= 500
FREQ ?= 5
STAT_FREQ ?= 150
SAVE_FREQ ?= 5000
PREFETCH ?= True
SCHEDULER_ARGS ?= '--scheduler ExpLR --lr 1e-3 --exp_step_size 50000 --exp_gamma 0.7'
OPTIMIZER_ARGS ?= '--optimizer Adam --bn_momentum 0.1'

#EXPORT_FEAT_MODE := sum
EXPORT_FEAT_MODE ?= max
#EXPORT_FEAT_MODE := avg

PS_MODEL_DATASET ?= BuildnetVoxelization0_01Dataset
AE_MODEL_DATASET ?= StylenetXYZAEVoxelization0_01Dataset
#AE_MODEL_DATASET ?= StylenetXYZAEVoxelization0_02Dataset
CLS_RNV_MODEL_DATASET ?= Stylenet_RNV_Voxelization0_01Dataset
#CLS_RNV_MODEL_DATASET ?= Stylenet_ROV_Voxelization0_01Dataset
CLS_MODEL_DATASET ?= StylenetComponentVoxelization0_01Dataset
LOSS_FACTOR := 100.
INPUT_FEAT ?= xyz_normals
IGNORE_LABEL ?= 255

train-AE:
	export DATASET=${AE_MODEL_DATASET} \
		&& export BATCH_SIZE=${BATCH_SIZE} \
		&& export MAX_ITER=${MAX_ITER} \
		&& export VAL_FREQ=${FREQ} \
		&& export STAT_FREQ=${STAT_FREQ} \
		&& export SAVE_FREQ=${SAVE_FREQ} \
		&& export SPLIT_TO_VAL=${SPLIT_TO_VAL} \
		&& export MODEL=${AE_MODEL} \
		&& export INPUT_FEAT=coords \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& export LOSS_FACTOR=${LOSS_FACTOR} \
		&& export PREFETCH_DATA=${PREFETCH} \
		&& export SCHEDULER_ARGS="${SCHEDULER_ARGS}" \
		&& export OPTIMIZER_ARGS="${OPTIMIZER_ARGS}" \
		&& ../../train.sh \
				${AE_DIR}/${AE_MODEL_DATASET}/AE-${AE_MODEL} \
				${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${TRAIN_SPLIT}
test-AE:
	export DATASET=${AE_MODEL_DATASET} \
		&& export STAT_FREQ=${STAT_FREQ} \
		&& export MODEL=${AE_MODEL} \
		&& export INPUT_FEAT=coords \
		&& export SPLIT_TO_RUN=test \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../../test.sh \
				${AE_DIR}/${AE_MODEL_DATASET}/AE-${AE_MODEL}-MultiGpu/${TEST_MODEL_DIR} \
				checkpoint_${AE_MODEL}best_loss.pth \
				${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${TEST_SPLIT}

export-features-AE:
	export DATASET=${AE_MODEL_DATASET} \
		&& export STAT_FREQ=${STAT_FREQ} \
		&& export MODEL=${AE_MODEL} \
		&& export INPUT_FEAT=coords \
		&& export SPLIT_TO_RUN=test \
		&& export PREFETCH_DATA=True \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../../export_features.sh \
				${AE_DIR}/${AE_MODEL_DATASET}/AE-${AE_MODEL}-MultiGpu/${TEST_MODEL_DIR} \
				checkpoint_${AE_MODEL}best_loss.pth \
				${LOGS_ROOT}/${TEST_SPLIT_DIR}/${TEST_SPLIT} \
				${TEST_DATA_NAME}

train-PS:
	export DATASET=${PS_MODEL_DATASET} \
		&& export BATCH_SIZE=${BATCH_SIZE} \
		&& export MAX_ITER=${MAX_ITER} \
		&& export VAL_FREQ=${FREQ} \
		&& export STAT_FREQ=${FREQ} \
		&& export TEST_STAT_FREQ=${TEST_STAT_FREQ} \
		&& export SAVE_FREQ=${SAVE_FREQ} \
		&& export SPLIT_TO_VAL=${SPLIT_TO_VAL} \
		&& export DATA_PATH_ARGS="--buildnet_path ${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${TRAIN_SPLIT}" \
		&& export MODEL=${PS_MODEL} \
		&& export INPUT_FEAT=normals \
		&& export PREFETCH_DATA=${PREFETCH} \
		&& export SCHEDULER_ARGS=${SCHEDULER_ARGS} \
		&& export OPTIMIZER_ARGS=${OPTIMIZER_ARGS} \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../../train.sh \
				${PS_DIR}/${PS_MODEL_DATASET}/PS-${PS_MODEL}


export-features-PS:
	export DATASET=$(PS_MODEL_DATASET) \
		&& export STAT_FREQ=$(STAT_FREQ) \
		&& export MODEL=$(PS_MODEL) \
		&& export INPUT_FEAT=normals \
		&& export SPLIT_TO_RUN=test \
		&& export PREFETCH_DATA=True \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../../export_features.sh \
				$(PS_DIR)/$(PS_MODEL_DATASET)/PS-$(PS_MODEL)/$(TEST_MODEL_DIR) \
				checkpoint_$(PS_MODEL)best_acc.pth \
				$(LOGS_ROOT)/$(TEST_SPLIT_DIR)/$(TEST_SPLIT) \
				$(TEST_DATA_NAME)
export-features-color-PS:
	export DATASET=$(PS_MODEL_DATASET) \
		&& export STAT_FREQ=$(STAT_FREQ) \
		&& export MODEL=$(PS_MODEL) \
		&& export INPUT_FEAT=normals_rgb \
		&& export SPLIT_TO_RUN=test \
		&& export PREFETCH_DATA=True \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../../export_features.sh \
				$(PS_DIR)/$(PS_MODEL_DATASET)/PS-Color-$(PS_MODEL)/$(TEST_MODEL_DIR) \
				checkpoint_$(PS_MODEL)best_acc.pth \
				$(LOGS_ROOT)/$(TEST_SPLIT_DIR)/$(TEST_SPLIT) \
				$(TEST_DATA_NAME)


train-CLS-RNV:
	export DATASET=${CLS_RNV_MODEL_DATASET} \
		&& export BATCH_SIZE=${BATCH_SIZE} \
		&& export MAX_ITER=${MAX_ITER} \
		&& export VAL_FREQ=${FREQ} \
		&& export STAT_FREQ=${STAT_FREQ} \
		&& export SAVE_FREQ=${SAVE_FREQ} \
		&& export SPLIT_TO_VAL=${SPLIT_TO_VAL} \
		&& export MODEL=${CLS_RNV_MODEL} \
		&& export INPUT_FEAT=normals \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& export LOSS_FACTOR=${LOSS_FACTOR} \
		&& export PREFETCH_DATA=${PREFETCH} \
		&& export SCHEDULER_ARGS=${SCHEDULER_ARGS} \
		&& export OPTIMIZER_ARGS=${OPTIMIZER_ARGS} \
		&& ../../train.sh \
				${CLS_RNV_DIR}/${FOLD}/${CLS_RNV_MODEL_DATASET}/CLS-RNV-${CLS_RNV_MODEL} \
				${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${FOLD}/${TRAIN_SPLIT}

train-CLS-style:
	export DATASET=$(CLS_MODEL_DATASET) \
		&& export BATCH_SIZE=$(BATCH_SIZE) \
		&& export MAX_ITER=$(MAX_ITER) \
		&& export VAL_FREQ=$(FREQ) \
		&& export STAT_FREQ=$(STAT_FREQ) \
		&& export SAVE_FREQ=$(SAVE_FREQ) \
		&& export SPLIT_TO_VAL=$(SPLIT_TO_VAL) \
		&& export MODEL=$(CLS_MODEL) \
		&& export INPUT_FEAT=$(INPUT_FEAT) \
		&& export IGNORE_LABEL=$(IGNORE_LABEL) \
		&& export DATA_PATH_ARGS="--stylenet_path $(LOGS_ROOT)/$(SPLITS_DIR)/$(PLY_SAMPLES)/$(TRAIN_SPLIT)" \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& export PREFETCH_DATA=$(PREFETCH) \
		&& export SCHEDULER_ARGS=$(SCHEDULER_ARGS) \
		&& export OPTIMIZER_ARGS=$(OPTIMIZER_ARGS) \
		&& ../../train.sh \
				$(CLS_STYLE_DIR)/$(CLS_MODEL_DATASET)/CLS-STYLE-$(CLS_MODEL)
