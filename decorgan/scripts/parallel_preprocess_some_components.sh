#!/bin/bash

SOURCE_DIR=${SOURCE_DIR:-"/home/maverkiou/zavou/DECOR-GAN"}
PY_EXE=${PY_EXE:-"/home/maverkiou/miniconda2/envs/decorgan/bin/python"}
CLASS_ID=${CLASS_ID:-"buildnet_component"}
SOURCE_ROOT=${SOURCE_ROOT:-"/mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/data"}
TARGET_DIR=${TARGET_DIR:-"/mnt/nfs/work1/kalo/maverkiou/zavou/decorgan-logs/preprocessed_data"}
UNIQUE_DIR=${UNIQUE_DIR:-"unique_point_clouds"}
PARTITION=${PARTITION:-m40-long}
TOTAL_JOBS=${TOTAL_JOBS:-6}
ONLY_UNIQUE=${ONLY_UNIQUE:-True}
BUILDING_CSV=${BUILDING_CSV:-""}
DISPLAY=${DISPLAY:-"50"}

export SOURCE_DIR=${SOURCE_DIR}
export PY_EXE=${PY_EXE}
export CLASS_ID=${CLASS_ID}
export SOURCE_ROOT=${SOURCE_ROOT}
export TARGET_DIR=${TARGET_DIR}
export UNIQUE_DIR=${UNIQUE_DIR}
export ONLY_UNIQUE=${ONLY_UNIQUE}
export BUILDING_CSV=${BUILDING_CSV}
export DISPLAY=${DISPLAY}


run_job(){
  job_num=$1
  if [ -d "/home/maverkiou" ] | [ -d "/home/czavou01" ]
  then
    echo "on cluster"
    echo "--job-name=prepsome_${job_num} --partition=${PARTITION} preprocess_some_components.sh $job_num ${TOTAL_JOBS}"
    sbatch --job-name=prepsome_${job_num} --partition=${PARTITION} preprocess_some_components.sh $job_num ${TOTAL_JOBS}
  else
    echo "on local. please be careful when stopping the job ... you might need to check ps -aux | grep preprocess and kill manually all processes"
#    sh ./preprocess_some_components.sh ${job_num} ${TOTAL_JOBS}
    sh ./preprocess_some_components.sh ${job_num} ${TOTAL_JOBS} &
  fi

}


i=0
while [ "$i" -lt "$TOTAL_JOBS" ]; do
  run_job $i
  i=$(( i + 1 ))
done
