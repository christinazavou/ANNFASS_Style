#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err


SOURCE_DIR=${SOURCE_DIR:-"/home/maverkiou/zavou/DECOR-GAN"}
PY_EXE=${PY_EXE:-"/home/maverkiou/miniconda2/envs/decorgan/bin/python"}
CLASS_ID=${CLASS_ID:-"buildnet_component"}
SOURCE_ROOT=${SOURCE_ROOT:-"/mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/data"}
TARGET_DIR=${TARGET_DIR:-"/mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/preprocessed_data"}
DISPLAY=${DISPLAY:-"50"}

echo "SOURCE_DIR: ${SOURCE_DIR}"
echo "PY_EXE: ${PY_EXE}"
echo "CLASS_ID: ${CLASS_ID}"
echo "SOURCE_ROOT: ${SOURCE_ROOT}"
echo "TARGET_DIR: ${TARGET_DIR}"
echo "DISPLAY: ${DISPLAY}"

PROCESS_ID=$1
TOTAL_PROCESSES=$2

echo "cd ${SOURCE_DIR}/data_preparation/hu_yu_lun_buildnet"
cd ${SOURCE_DIR}/data_preparation/hu_yu_lun_buildnet

log_file="${SOURCE_DIR}/data_preparation/hu_yu_lun_buildnet/${PROCESS_ID}_${TOTAL_PROCESSES}.log"


if [ -d "/home/czavou01" ]
  then
    echo "module load Xvfb/1.20.9-GCCcore-10.2.0"
    module load Xvfb/1.20.9-GCCcore-10.2.0
    echo "module load libGLU/9.0.1-GCCcore-10.2.0"
    module load libGLU/9.0.1-GCCcore-10.2.0
    echo "Xvfb :${DISPLAY} &"
    Xvfb :${DISPLAY} &
    echo "export DISPLAY=:${DISPLAY}"
    export DISPLAY=:${DISPLAY}
  fi

args="${CLASS_ID} ${SOURCE_ROOT} ${TARGET_DIR} ${PROCESS_ID} ${TOTAL_PROCESSES}"
echo "${PY_EXE} preprocess.py $args >> $log_file"
${PY_EXE} preprocess.py $args >> $log_file

