SHELL := /bin/bash

PY_EXE := /home/graphicslab/miniconda3/envs/py3-mink/bin/python
STYLE_ROOT := /media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection
LOGS_ROOT := ${STYLE_ROOT}/logs

DATA_NAME ?= annfass
PLY_SAMPLES ?= ply_100K_cnscr
SPLITS_SUFFIX ?= splits_march
TRAIN_SPLIT ?= split_train_test
SPLIT_TO_VAL ?= test
TEST_SPLIT ?= split_test
FOLD ?= fold0
TEST_MODEL_DIR ?= b5-i500
TEST_DATA_NAME ?= annfasscomponent
TEST_SPLIT_DIR ?= annfass_splits_march/ply_100K_cnscr/fold0

SPLITS_DIR := ${DATA_NAME}_${SPLITS_SUFFIX}
AE_DIR := ${LOGS_ROOT}/${DATA_NAME}_minkowski_ae/${DATA_NAME}_${PLY_SAMPLES}
CLS_DIR := ${LOGS_ROOT}/${DATA_NAME}_minkowski_cls/${DATA_NAME}_${PLY_SAMPLES}
CLS_RNV_DIR := ${LOGS_ROOT}/${DATA_NAME}_minkowski_cls_rnv/${DATA_NAME}_${PLY_SAMPLES}

#AE_MODEL := Res16UNetAEL8B2
#AE_MODEL := HRNetAE3S2BD128
AE_MODEL ?= HRNetAE3S2BD256
#AE_MODEL ?= HRNetAE3S3BD64IN8
#AE_MODEL ?= HRNetAE3S3BD64IN16
#AE_MODEL ?= HRNetAE3S3BND64IN16
#AE_MODEL := SmallNetAE
#AE_MODEL := HRNetAE1S2BD128
#AE_MODEL := HRNetAE3S3BND256
#AE_MODEL := HRNetAE3S2BND128
#CLS_MODEL ?= SmallNetStyleCls
CLS_MODEL ?= HRNetStyleCls3S2BND128
CLS_RNV_MODEL ?= HRNet3S2BD256
BATCH_SIZE ?= 32
MAX_ITER ?= 50000
FREQ ?= 50
STAT_FREQ ?= 150
SAVE_FREQ ?= 5000
PREFETCH ?= True
SCHEDULER_ARGS ?= '--scheduler ExpLR --lr 1e-3 --exp_step_size 50000 --exp_gamma 0.7'
OPTIMIZER_ARGS ?= '--optimizer Adam --bn_momentum 0.1'

#EXPORT_FEAT_MODE := sum
EXPORT_FEAT_MODE ?= max
#EXPORT_FEAT_MODE := avg

AE_MODEL_DATASET ?= StylenetXYZAEVoxelization0_01Dataset
#AE_MODEL_DATASET ?= StylenetXYZAEVoxelization0_02Dataset
CLS_MODEL_DATASET ?= StylenetComponentVoxelization0_01Dataset
CLS_RNV_MODEL_DATASET ?= Stylenet_RNV_Voxelization0_01Dataset
#CLS_RNV_MODEL_DATASET ?= Stylenet_ROV_Voxelization0_01Dataset
LOSS_FACTOR := 100.


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
				${AE_DIR}/${FOLD}/${AE_MODEL_DATASET}/AE-${AE_MODEL} \
				${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${FOLD}/${TRAIN_SPLIT}
train-CLS:
	export DATASET=${CLS_MODEL_DATASET} \
		&& export BATCH_SIZE=${BATCH_SIZE} \
		&& export MAX_ITER=${MAX_ITER} \
		&& export VAL_FREQ=${FREQ} \
		&& export STAT_FREQ=${STAT_FREQ} \
		&& export SAVE_FREQ=${SAVE_FREQ} \
		&& export SPLIT_TO_VAL=${SPLIT_TO_VAL} \
		&& export MODEL=${CLS_MODEL} \
		&& export INPUT_FEAT=normals \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& export LOSS_FACTOR=${LOSS_FACTOR} \
		&& export PREFETCH_DATA=${PREFETCH} \
		&& export SCHEDULER_ARGS=${SCHEDULER_ARGS} \
		&& export OPTIMIZER_ARGS=${OPTIMIZER_ARGS} \
		&& ../../train.sh \
				${CLS_DIR}/${FOLD}/${CLS_MODEL_DATASET}/CLS-${CLS_MODEL} \
				${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${FOLD}/${TRAIN_SPLIT}
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
#test-AE:
#	export DATASET=${AE_MODEL_DATASET} \
#		&& export STAT_FREQ=${STAT_FREQ} \
#		&& export MODEL=${AE_MODEL} \
#		&& export INPUT_FEAT=coords \
#		&& export SPLIT_TO_RUN=test \
#		&& export PREFETCH_DATA=True \
#		&& export CUDA_VISIBLE_DEVICES=0 \
#		&& ../../test.sh \
#				${AE_DIR}/${FOLD}/${AE_MODEL_DATASET}/AE-${AE_MODEL}/${TEST_MODEL_DIR} \
#				checkpoint_${AE_MODEL}best_loss.pth \
#				${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${FOLD}/${TEST_SPLIT}
test-CLS:
	export DATASET=${CLS_MODEL_DATASET} \
		&& export STAT_FREQ=${STAT_FREQ} \
		&& export MODEL=${CLS_MODEL} \
		&& export INPUT_FEAT=normals \
		&& export SPLIT_TO_RUN=test \
		&& export PREFETCH_DATA=${PREFETCH} \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../../test.sh \
				${CLS_DIR}/${FOLD}/${CLS_MODEL_DATASET}/CLS-${CLS_MODEL}/${TEST_MODEL_DIR} \
				checkpoint_${CLS_MODEL}best_loss.pth \
				${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${FOLD}/${TEST_SPLIT}
test-AE-otherdata:
	export DATASET=${AE_MODEL_DATASET} \
		&& export STAT_FREQ=${STAT_FREQ} \
		&& export MODEL=${AE_MODEL} \
		&& export INPUT_FEAT=coords \
		&& export SPLIT_TO_RUN=test \
		&& export PREFETCH_DATA=True \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& ../../test.sh \
				${AE_DIR}/${FOLD}/${AE_MODEL_DATASET}/AE-${AE_MODEL}/${TEST_MODEL_DIR} \
				checkpoint_${AE_MODEL}best_loss.pth \
				${LOGS_ROOT}/${TEST_SPLIT_DIR}/${TEST_SPLIT}
#export-features-AE:
#	export DATASET=${AE_MODEL_DATASET} \
#		&& export STAT_FREQ=${STAT_FREQ} \
#		&& export MODEL=${AE_MODEL} \
#		&& export INPUT_FEAT=coords \
#		&& export SPLIT_TO_RUN=test \
#		&& export PREFETCH_DATA=True \
#		&& export CUDA_VISIBLE_DEVICES=0 \
#		&& export EXPORT_FEAT_MODE=${EXPORT_FEAT_MODE} \
#		&& ../../export_features.sh \
#				${AE_DIR}/${FOLD}/${AE_MODEL_DATASET}/AE-${AE_MODEL}/${TEST_MODEL_DIR} \
#				checkpoint_${AE_MODEL}best_loss.pth \
#				${LOGS_ROOT}/${SPLITS_DIR}/${PLY_SAMPLES}/${FOLD}/${TEST_SPLIT} \
#				${DATA_NAME}
export-features-AE-otherdata:
	export DATASET=${AE_MODEL_DATASET} \
		&& export STAT_FREQ=${STAT_FREQ} \
		&& export MODEL=${AE_MODEL} \
		&& export INPUT_FEAT=coords \
		&& export SPLIT_TO_RUN=test \
		&& export PREFETCH_DATA=True \
		&& export CUDA_VISIBLE_DEVICES=0 \
		&& export EXPORT_FEAT_MODE=${EXPORT_FEAT_MODE} \
		&& ../../export_features.sh \
				${AE_DIR}/${FOLD}/${AE_MODEL_DATASET}/AE-${AE_MODEL}/${TEST_MODEL_DIR} \
				checkpoint_${AE_MODEL}best_loss.pth \
				${LOGS_ROOT}/${TEST_SPLIT_DIR}/${TEST_SPLIT} \
				${TEST_DATA_NAME}
