#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=titanx-long    # Partition to submit to
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=10000   # Memory in MB per cpu allocated

TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export PYTHONUNBUFFERED="True"

LOG_DIR=$1
echo "LogDir $LOG_DIR"

if [ -d /home/maverkiou/miniconda2/envs/py3-mink/bin ]; then
  PY_EXE=/home/maverkiou/miniconda2/envs/py3-mink/bin/python
  SOURCE_DIR=/home/maverkiou/zavou/style_detection/minkoski_pytorch
else
  PY_EXE=/home/graphicslab/miniconda3/envs/py3-mink/bin/python
  SOURCE_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/minkoski_pytorch
fi

DATASET=${DATASET:-StylenetAEVoxelization0_01Dataset}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ITER=${MAX_ITER:-100}
VAL_FREQ=${VAL_FREQ:-20}
STAT_FREQ=${STAT_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-100}
PREVIOUS_CHECKPOINT=${PREVIOUS_CHECKPOINT:-No}
SPLIT_TO_TRAIN=${SPLIT_TO_TRAIN:-train}
SPLIT_TO_VAL=${SPLIT_TO_VAL:-val}
NORMALIZE_Y=${NORMALIZE_Y:-False}
MODEL=${MODEL:-HRNetAE3S2BD128}
INPUT_FEAT=${INPUT_FEAT:-coords}
MULTI_GPU=${MULTI_GPU:-False}
LOSS_FACTOR=${LOSS_FACTOR:-10.}
PREFETCH_DATA=${PREFETCH_DATA:-False}
IGNORE_LABEL=${IGNORE_LABEL:-255}
TEST_STAT_FREQ=${TEST_STAT_FREQ:-100}
SCHEDULER_ARGS=${SCHEDULER_ARGS:-"--scheduler ReduceLROnPlateau --lr 1e-2"}
OPTIMIZER_ARGS=${OPTIMIZER_ARGS:-"--optimizer SGD"}
DATA_PATH_ARGS=${DATA_PATH_ARGS:-"--vaebcecomponent_path ModelNet40chair"}
#DATA_PATH_ARGS=${DATA_PATH_ARGS:-"--buildnet_path ModelNet40chair"}
#DATA_PATH_ARGS=${DATA_PATH_ARGS:-"--stylenet_path ModelNet40chair"}
LOSS_ARGS=${LOSS_ARGS:-"--chamfer_loss True"}

LOG_DIR=${LOG_DIR}/b${BATCH_SIZE}-i${MAX_ITER}

if [ -d "${LOG_DIR}" ]; then
  echo "log dir exists. will generate another one"
  LOG_DIR="${LOG_DIR}_${TIME}"
fi
mkdir -p $LOG_DIR

const="\
--train_limit_numpoints 1200000 \
--normalize_color False \
--normalize_coords True"

args="$const"
args+=" --model ${MODEL}"
args+=" --dataset ${DATASET}"
args+=" --log_dir $LOG_DIR"
args+=" --batch_size ${BATCH_SIZE}"
args+=" --max_iter ${MAX_ITER}"
args+=" --input_feat ${INPUT_FEAT}"
args+=" --val_freq ${VAL_FREQ}"
args+=" --stat_freq ${STAT_FREQ}"
args+=" --test_stat_freq ${TEST_STAT_FREQ}"
args+=" --save_freq ${SAVE_FREQ}"
args+=" --train_phase ${SPLIT_TO_TRAIN}"
args+=" --val_phase ${SPLIT_TO_VAL}"
args+=" --normalize_y ${NORMALIZE_Y}"
args+=" --loss_factor ${LOSS_FACTOR}"
args+=" --multi_gpu ${MULTI_GPU}"
args+=" --prefetch_data ${PREFETCH_DATA}"
args+=" --ignore_label ${IGNORE_LABEL}"
args+=" $SCHEDULER_ARGS"
args+=" $OPTIMIZER_ARGS"
args+=" $LOSS_ARGS"
args+=" $DATA_PATH_ARGS"

if [ "${PREVIOUS_CHECKPOINT}" != "No" ]; then
  args+=" --resume ${PREVIOUS_CHECKPOINT}"
  args+=" --resume_optimizer True"
fi

VERSION=$(git rev-parse HEAD)

LOG="$LOG_DIR/$TIME.txt"

echo Logging output to "$LOG"
echo "Version: ${VERSION}" > "$LOG"
echo -e "GPU(s): $CUDA_VISIBLE_DEVICES" >> $LOG
echo "cd ${SOURCE_DIR} && ${PY_EXE} main.py $args" >> "$LOG"
cd ${SOURCE_DIR} && ${PY_EXE} main.py $args 2>&1 | tee -a "$LOG"
