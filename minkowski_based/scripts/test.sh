#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

PY_EXE=/home/graphicslab/miniconda3/envs/py3-mink/bin/python
SOURCE_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/minkoski_pytorch

TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export PYTHONUNBUFFERED="True"

MODEL_DIR=$1
MODEL_WEIGHTS=$2
INPUT_DATA=$3

DATASET=${DATASET:-StylenetAEVoxelization0_01Dataset}
STAT_FREQ=${STAT_FREQ:-1}
SPLIT_TO_RUN=${SPLIT_TO_RUN:-test}
NORMALIZE_Y=${NORMALIZE_Y:-False}
MODEL=${MODEL:-Res16UNetStyleAE}
LOSS_ARGS=${LOSS_ARGS:-"--chamfer_loss True"}
INPUT_FEAT=${INPUT_FEAT:-coords}


LOG_DIR=${MODEL_DIR}/evaluation
if [ -d "${LOG_DIR}" ]; then
  echo "log dir exists. will generate another one"
  LOG_DIR="${LOG_DIR}_${TIME}"
fi
mkdir -p $LOG_DIR

const="\
--train_limit_numpoints 1200000 \
--normalize_color False \
--normalize_coords True \
--is_train false \
--test_original_pointcloud true \
--return_transformation true \
--save_prediction true \
--idw true \
--multi_gpu False"

args="$const"
args+=" --model ${MODEL}"
args+=" --dataset ${DATASET}"
args+=" --log_dir $LOG_DIR"
args+=" --input_feat ${INPUT_FEAT}"
args+=" --save_pred_dir ${LOG_DIR}/predict_on_${SPLIT_TO_RUN}"
args+=" --weights ${MODEL_DIR}/${MODEL_WEIGHTS}"
args+=" --test_stat_freq ${STAT_FREQ}"
args+=" --test_phase ${SPLIT_TO_RUN}"
args+=" --normalize_y ${NORMALIZE_Y}"
args+=" $LOSS_ARGS"

dataclass="Build"
if [[ $DATASET == *"$dataclass"* ]]; then
  args+=" --buildnet_path ${INPUT_DATA}"
else
  args+=" --stylenet_path ${INPUT_DATA}"
fi

VERSION=$(git rev-parse HEAD)

LOG="$LOG_DIR/$TIME.txt"

echo Logging output to "$LOG"
echo "Version: " $VERSION >> $LOG
echo -e "GPU(s): $CUDA_VISIBLE_DEVICES" >> $LOG
echo "cd ${SOURCE_DIR} && ${PY_EXE} main.py $args" >> "$LOG"
cd ${SOURCE_DIR} && ${PY_EXE} main.py $args 2>&1 | tee -a "$LOG"

#--prefetch_data true \
