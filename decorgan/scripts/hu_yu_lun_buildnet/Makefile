TURING_SOURCE_DIR ?= /home/czavou01/decor-gan-private
TURING_PY_EXE ?= /home/czavou01/miniconda3/envs/decorgan/bin/python
TURING_LOGS_DIR ?= /home/czavou01/decorgan-logs
TURING_MODELS_DIR ?= /home/czavou01/decorgan-models
TURING_DATA_ROOT ?= /home/czavou01/ANNFASS_DATA
PARTITION ?= COMPUTE
TOTAL_JOBS ?= 30
DISPLAY ?= 95
# ----------------------------------------------------------------------------------------------------------------

shapenet_chairs_decor_preprocess_random_rot:  # check num of cores with lscpu
	export SOURCE_DIR=$(TURING_SOURCE_DIR) \
		&& export PY_EXE=$(TURING_PY_EXE) \
		&& export SOURCE_ROOT=/mnt/beegfs/czavou01/decorgan-logs \
		&& export TARGET_DIR=/mnt/beegfs/czavou01/decorgan-logs/preprocessed_data \
		&& export CLASS_ID=03001627_train_objs_norm_and_random_rot \
		&& export PARTITION=$(PARTITION) \
		&& export TOTAL_JOBS=$(TOTAL_JOBS) \
		&& export DISPLAY=$(DISPLAY) \
		&& sh ./parallel_preprocess.sh

#class ?= 03001627
class ?= 02958343
shapenet_class_decor_preprocess_specific_rot:  # check num of cores with lscpu
	export SOURCE_DIR=$(TURING_SOURCE_DIR) \
		&& export PY_EXE=$(TURING_PY_EXE) \
		&& export SOURCE_ROOT=/mnt/beegfs/czavou01/decorgan-logs \
		&& export TARGET_DIR=/mnt/beegfs/czavou01/decorgan-logs/preprocessed_data \
		&& export CLASS_ID=$(class)_train_objs_norm_and_specific_rot \
		&& export PARTITION=$(PARTITION) \
		&& export TOTAL_JOBS=$(TOTAL_JOBS) \
		&& export DISPLAY=$(DISPLAY) \
		&& sh ./parallel_preprocess.sh

shapenet_class_test_decor_preprocess_specific_rot:  # check num of cores with lscpu
	export SOURCE_DIR=$(TURING_SOURCE_DIR) \
		&& export PY_EXE=$(TURING_PY_EXE) \
		&& export SOURCE_ROOT=/mnt/beegfs/czavou01/decorgan-logs \
		&& export TARGET_DIR=/mnt/beegfs/czavou01/decorgan-logs/preprocessed_data \
		&& export CLASS_ID=$(class)_test_objs_norm_and_specific_rot \
		&& export PARTITION=$(PARTITION) \
		&& export TOTAL_JOBS=$(TOTAL_JOBS) \
		&& export DISPLAY=$(DISPLAY) \
		&& sh ./parallel_preprocess.sh
