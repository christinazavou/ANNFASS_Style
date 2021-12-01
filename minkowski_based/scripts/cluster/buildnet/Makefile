SHELL := /bin/bash

SPLIT_TO_VAL := val
PARTITION ?= titanx-long
PS_MODEL ?= Res16UNet34A
#PS_MODEL ?= HRNet3S2BD256
BATCH_SIZE ?= 32
MAX_ITER ?= 100000
FREQ ?= 100  # if 1500 train data and batch 32
SAVE_FREQ ?= 10000
NUM_GPUS ?= 4
PREFETCH ?= True
TEST_STAT_FREQ := 1000
SCHEDULER_ARGS ?= '--scheduler ExpLR --lr 1e-3 --exp_step_size 50000 --exp_gamma 0.7'
OPTIMIZER_ARGS ?= '--optimizer Adam --bn_momentum 0.1'
PS_MODEL_DATASET ?= BuildnetVoxelization0_01Dataset
BUILDNET_PATH = /mnt/nfs/work1/kalo/maverkiou/zavou/data/mink_splits/ps_on_buildnet
RESULT_DIR = /mnt/nfs/work1/kalo/maverkiou/zavou/mink_results/ps_on_buildnet

train-PS:
	export DATASET=$(PS_MODEL_DATASET) \
		&& export BATCH_SIZE=$(BATCH_SIZE) \
		&& export MAX_ITER=$(MAX_ITER) \
		&& export VAL_FREQ=$(FREQ) \
		&& export STAT_FREQ=$(FREQ) \
		&& export TEST_STAT_FREQ=$(TEST_STAT_FREQ) \
		&& export SAVE_FREQ=$(SAVE_FREQ) \
		&& export SPLIT_TO_VAL=$(SPLIT_TO_VAL) \
		&& export DATA_PATH_ARGS="--buildnet_path $(BUILDNET_PATH)" \
		&& export MODEL=$(PS_MODEL) \
		&& export INPUT_FEAT=normals \
		&& export PREFETCH_DATA=$(PREFETCH) \
		&& export SCHEDULER_ARGS="$(SCHEDULER_ARGS)" \
		&& export OPTIMIZER_ARGS="$(OPTIMIZER_ARGS)" \
		&& sbatch \
			--job-name=PSonB \
			--partition=$(PARTITION) \
			../../train.sh \
				$(RESULT_DIR)/$(PS_MODEL_DATASET)/PS-$(PS_MODEL)

train-PS-MultiGpu:
	export DATASET=$(PS_MODEL_DATASET) \
		&& export BATCH_SIZE=$(BATCH_SIZE) \
		&& export MAX_ITER=$(MAX_ITER) \
		&& export VAL_FREQ=$(FREQ) \
		&& export STAT_FREQ=$(FREQ) \
		&& export TEST_STAT_FREQ=$(TEST_STAT_FREQ) \
		&& export SAVE_FREQ=$(SAVE_FREQ) \
		&& export SPLIT_TO_VAL=$(SPLIT_TO_VAL) \
		&& export DATA_PATH_ARGS="--buildnet_path $(BUILDNET_PATH)" \
		&& export MODEL=$(PS_MODEL) \
		&& export MULTI_GPU=True \
		&& export INPUT_FEAT=normals \
		&& export PREFETCH_DATA=$(PREFETCH) \
		&& export SCHEDULER_ARGS="$(SCHEDULER_ARGS)" \
		&& export OPTIMIZER_ARGS="$(OPTIMIZER_ARGS)" \
		&& sbatch \
			--job-name=PSonB-multi \
			--gres=gpu:$(NUM_GPUS) \
			--partition=$(PARTITION) \
			../../train.sh \
				$(RESULT_DIR)/$(PS_MODEL_DATASET)/PS-$(PS_MODEL)

train-PS-Color-MultiGpu:
	export DATASET=$(PS_MODEL_DATASET) \
		&& export BATCH_SIZE=$(BATCH_SIZE) \
		&& export MAX_ITER=$(MAX_ITER) \
		&& export VAL_FREQ=$(FREQ) \
		&& export STAT_FREQ=$(FREQ) \
		&& export TEST_STAT_FREQ=$(TEST_STAT_FREQ) \
		&& export SAVE_FREQ=$(SAVE_FREQ) \
		&& export SPLIT_TO_VAL=$(SPLIT_TO_VAL) \
		&& export DATA_PATH_ARGS="--buildnet_path $(BUILDNET_PATH)" \
		&& export MODEL=$(PS_MODEL) \
		&& export MULTI_GPU=True \
		&& export INPUT_FEAT=normals_rgb \
		&& export PREFETCH_DATA=$(PREFETCH) \
		&& export SCHEDULER_ARGS="$(SCHEDULER_ARGS)" \
		&& export OPTIMIZER_ARGS="$(OPTIMIZER_ARGS)" \
		&& sbatch \
			--job-name=PSonB-color \
			--gres=gpu:$(NUM_GPUS) \
			--partition=$(PARTITION) \
			../../train.sh \
				$(RESULT_DIR)/$(PS_MODEL_DATASET)/PS-Color-$(PS_MODEL)
