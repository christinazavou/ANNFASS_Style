GYPSUM_SOURCE_DIR ?= /home/maverkiou/zavou/style_detection/minkoski_pytorch/vae
GYPSUM_PY_EXE ?= /home/maverkiou/miniconda2/envs/py3-mink/bin/python
LOCAL_SOURCE_DIR ?= /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/minkoski_pytorch/vae
LOCAL_PY_EXE ?= /home/graphicslab/miniconda3/envs/py3-mink/bin/python


#---------------------------------------original on buildnet----------------------------------------------------
train-vae-all-buildnet-gypsum:
	export SOURCE_DIR=$(GYPSUM_SOURCE_DIR) \
		&& export PY_EXE=$(GYPSUM_PY_EXE) \
		&& export LOG_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/mink_results/vae_on_buildnet_all \
		&& export DATASET=ComponentObjDataset \
		&& export DATA_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data/decorgan-preprocessed/groups_june17_uni_nor_components \
		&& export TRAIN_SPLIT_FILE=/mnt/nfs/work1/kalo/maverkiou/zavou/data/all_vae_split.txt \
		&& export SAVE_FREQ=6000 \
		&& export STAT_FREQ=600 \
		&& export VAL_FREQ=1200 \
		&& export MAX_ITER=60000 \
		&& export LR=1e-2 \
		&& sbatch --job-name=vaeall --partition=titanx-long train.sh
train-vae-train-buildnet-gypsum:
	export SOURCE_DIR=$(GYPSUM_SOURCE_DIR) \
		&& export PY_EXE=$(GYPSUM_PY_EXE) \
		&& export LOG_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/mink_results/vae_on_buildnet_train \
		&& export DATASET=ComponentObjDataset \
		&& export DATA_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data/decorgan-preprocessed/groups_june17_uni_nor_components \
		&& export TRAIN_SPLIT_FILE=/mnt/nfs/work1/kalo/maverkiou/zavou/data/train_vae_split.txt \
		&& export SAVE_FREQ=6000 \
		&& export STAT_FREQ=600 \
		&& export VAL_FREQ=1200 \
		&& export MAX_ITER=60000 \
		&& export LR=1e-2 \
		&& sbatch --job-name=vae --partition=titanx-long train.sh
train-vae-balanced-buildnet-gypsum:
	export SOURCE_DIR=$(GYPSUM_SOURCE_DIR) \
		&& export PY_EXE=$(GYPSUM_PY_EXE) \
		&& export LOG_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/mink_results/vae_on_buildnet_balanced \
		&& export DATASET=ComponentObjDataset \
		&& export DATA_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data/decorgan-preprocessed/groups_june17_uni_nor_components \
		&& export TRAIN_SPLIT_FILE=/mnt/nfs/work1/kalo/maverkiou/zavou/data/balanced_vae_split.txt \
		&& export SAVE_FREQ=6000 \
		&& export STAT_FREQ=600 \
		&& export VAL_FREQ=1200 \
		&& export MAX_ITER=60000 \
		&& export LR=1e-2 \
		&& sbatch --job-name=vaebal --partition=titanx-long train.sh
train-vae-local:
	export SOURCE_DIR=$(LOCAL_SOURCE_DIR) \
		&& export PY_EXE=$(LOCAL_PY_EXE) \
		&& export LOG_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/mink_results/local/debug_vae_on_buildnet \
		&& export DATASET=ComponentObjDataset \
		&& export DATA_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components \
		&& export TRAIN_SPLIT_FILE=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/simple_mesh/groups_june17_uni_nor_components/unique/columndomedoorwindowtower/split_train_val_test/train.txt \
		&& export SAVE_FREQ=2 \
		&& export STAT_FREQ=2 \
		&& export VAL_FREQ=5 \
		&& export MAX_ITER=6 \
		&& export LR=1e-2 \
		&& sh ./train.sh

LOCAL_LOG_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/mink_results/gypsum/nov16/mink_results/vae_on_buildnet_balanced
LOCAL_DATA_DIR=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components
VAL_SPLIT_FILE=/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/export_files/minkowski_export_buildnet_annfass_component.txt
CKPT_ITER=54000
WEIGHTS_CKPT=$(LOCAL_LOG_DIR)/checkpoints/model_iter$(CKPT_ITER).pth
ENCODINGS_DIR=encodings_buildnet_and_annfass
visualize-vae-local:
	export SOURCE_DIR=$(LOCAL_SOURCE_DIR) \
		&& export PY_EXE=$(LOCAL_PY_EXE) \
		&& export LOG_DIR=$(LOCAL_LOG_DIR) \
		&& export DATASET=ComponentObjDataset \
		&& export DATA_DIR=$(LOCAL_DATA_DIR) \
		&& export VAL_SPLIT_FILE=$(VAL_SPLIT_FILE) \
		&& export WEIGHTS_CKPT=$(WEIGHTS_CKPT) \
		&& sh ./visualize.sh
export-vae-local:
	export SOURCE_DIR=$(LOCAL_SOURCE_DIR) \
		&& export PY_EXE=$(LOCAL_PY_EXE) \
		&& export LOG_DIR=$(LOCAL_LOG_DIR) \
		&& export DATASET=ComponentObjDataset \
		&& export DATA_DIR=$(LOCAL_DATA_DIR) \
		&& export VAL_SPLIT_FILE=$(VAL_SPLIT_FILE) \
		&& export WEIGHTS_CKPT=$(WEIGHTS_CKPT) \
		&& export ENCODINGS_DIR=$(ENCODINGS_DIR) \
		&& sh ./export.sh
