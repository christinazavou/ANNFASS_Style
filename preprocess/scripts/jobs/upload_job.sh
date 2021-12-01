#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=5-00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --mem=22GB
#SBATCH --cpus-per-task=4


if [ -d /mnt/nfs/work1/kalo/maverkiou/zavou/data ]; then
  ROOT_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data
  STYLE_DIR=/home/maverkiou/zavou/style_detection
  PREPROCESS_DIR=${STYLE_DIR}/preprocess
  PY_EXE=/home/maverkiou/miniconda2/envs/style_detect_env/bin/python
elif [ -d /media/christina/Data/ANNFASS_data ]; then
  ROOT_DIR=/media/christina/Data/ANNFASS_data
  STYLE_DIR=/media/christina/Data/ANNFASS_code/zavou-repos/style_detection
  PREPROCESS_DIR=${STYLE_DIR}/preprocess
  PY_EXE=/home/christina/miniconda3/envs/STYLE/bin/python
elif [ -d /media/graphicslab/BigData/zavou/ANNFASS_DATA ]; then
  ROOT_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA
  STYLE_DIR=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection
  PREPROCESS_DIR=${STYLE_DIR}/preprocess
  PY_EXE=/home/graphicslab/miniconda3/envs/style_detect_env/bin/python
fi


GROUPS_DIR_PREFIX="groups"
RENDERS_DIR_PREFIX="renderings"
VIEWS_DIR_PREFIX="viewpoints"

DATA_REPO=$1
CSV_FILE=$2
GDREPO=$3
echo "DATA_REPO ${DATA_REPO}"
echo "CSV_FILE ${CSV_FILE}"
echo "GDREPO ${GDREPO}"


args="--root_dir ${ROOT_DIR}/${DATA_REPO}"
args+=" --gdrepo ${GDREPO}"
args+=" --render_path $RENDERS_DIR_PREFIX"
args+=" --viewpoints_path $VIEWS_DIR_PREFIX"
args+=" --groups_path $GROUPS_DIR_PREFIX"
args+=" --buildings_csv $CSV_FILE"
args+=" --logs_dir ${ROOT_DIR}/${DATA_REPO}/upload_logs"
echo "Calling ${PY_EXE} upload.py $args"

cd ${PREPROCESS_DIR}/google_drive && \
    ${PY_EXE} upload.py $args && \
    rm $CSV_FILE
