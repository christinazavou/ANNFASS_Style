#!/bin/bash

# NOTES REGARDING BUILDIINGS.CSV:
# PLEASE DONT USE AS END_IDX A NUMBER BIGGER THAN THE AMOUNT OF LINES  !!!!!!!!!!!!!!
# PLEASE REMOVE ANY LINES REGARDING BUILDINGS THAT CANNOT BE FOUND IN RAW_DATA FOLDER


#if [ -z "$DATA_ROOT" ]; then
#  echo "Please provide DATA_ROOT argument"
#  exit 1
#fi


if [ -d /home/maverkiou/ ]; then
  ON_CLUSTER=1
elif [ -d /home/czavou01/ ]; then
  ON_CLUSTER=1
else
  ON_CLUSTER=0
fi
echo "DATA_ROOT: ${DATA_ROOT}"
echo "ON_CLUSTER: ${ON_CLUSTER}"


if [ -d "/home/czavou01" ]
  then
#    echo "module load Xvfb/1.20.9-GCCcore-10.2.0"
#    module load Xvfb/1.20.9-GCCcore-10.2.0
#    echo "module load libGLU/9.0.1-GCCcore-10.2.0"
#    module load libGLU/9.0.1-GCCcore-10.2.0
#    echo "Xvfb :50 &"
#    Xvfb :50 &
#    echo "export DISPLAY=:50"
#    export DISPLAY=:50
    GPU=${GPU:-0}
    export CUDA_VISIBLE_DEVICES=${GPU}
  fi


PY_EXE=${PY_EXE:="/home/maverkiou/miniconda2/envs/style_detect_env/bin/python"}
BLENDER_EXE=${BLENDER_EXE:="/home/maverkiou/blender-2.91.0-linux64/blender"}
STYLE_REPO=${STYLE_REPO:="/home/maverkiou/miniconda_3/envs/style_detect_env/bin/python"}
echo "PY_EXE: ${PY_EXE}"
echo "BLENDER_EXE: ${BLENDER_EXE}"
echo "STYLE_REPO: ${STYLE_REPO}"

NAME=${NAME:="COLOURCLOUD"}
JOB_FILE=${JOB_FILE:="colour_point_cloud_job.sh"}
USE_FUNCTION=${USE_FUNCTION:="run_coarse_detailed_plys"}
SBATCH_ARGS=${SBATCH_ARGS:="--cpus-per-task 8 --partition m40-long --mem 22GB --gres=gpu:1"}
START_IDX=${START_IDX:=0}
END_IDX=${END_IDX:=8}
STEP_SIZE=${STEP_SIZE:=4}
LOGS_FILE="$NAME.log"
echo "LOGS_FILE: $LOGS_FILE"
echo "JOB_FILE: $JOB_FILE"
echo "USE_FUNCTION: $USE_FUNCTION"
echo "START_IDX: $START_IDX"
echo "END_IDX: $END_IDX"
echo "STEP_SIZE: $STEP_SIZE"
echo "SBATCH_ARGS: $SBATCH_ARGS"


GOOGLE_DRIVE_REPO=${GOOGLE_DRIVE_REPO:="BUILDNET_Buildings"}
DATA_REPO=${DATA_REPO:="BUILDNET_Buildings"}
INIT_SAMPLES=${INIT_SAMPLES:="10000K"}
BUILDINGS_FILE_PREFIX=${BUILDINGS_FILE_PREFIX:="buildings"}
OBJ_DIR=${OBJ_DIR:="normalizedObj_refinedTextures"}
SAMPLE_POINTS_DIR=${SAMPLE_POINTS_DIR:="samplePoints_refinedTextures"}
PLY_DIR_PREFIX=${PLY_DIR_PREFIX:="ply"}
UNIQUE_DIR=${UNIQUE_DIR:="unique_point_clouds"}
GROUPS_DIR=${GROUPS_DIR:="groups"}
PLY_N_DIR=${PLY_N_DIR:="normalizedPly"}
VIEWPOINTS_DIR=${VIEWPOINTS_DIR:="viewpoints"}
RENDERINGS_DIR=${RENDERINGS_DIR:="renderings"}
OBJCOMPONENTS_DIR=${OBJCOMPONENTS_DIR:="unified_normalized_components"}
WITH_GROUPS=${WITH_GROUPS:="False"}
CUT_AT=${CUT_AT:=-1}
RIDGE_VALLEY_DIR=${RIDGE_VALLEY_DIR:="ridge_or_valley"}
COLOR_DIR=${COLOR_DIR:="color"}
NUM_PROCESSES=${NUM_PROCESSES:=8}
PER_COMPONENT=${PER_COMPONENT:="False"}
SAMPLES=${SAMPLES:=1000000}
EXPORT_CURVATURES=${EXPORT_CURVATURES:=False}
EXPORT_PCA=${EXPORT_PCA:=False}
OVERRIDE=${OVERRIDE:=False}
REMOVE=${REMOVE:=False}
PTS_ON_STYLE=${PTS_ON_STYLE:=False}
ON_GPU=${ON_GPU:=True}
RENDER_MODE=${RENDER_MODE:=2}
echo "DATA_REPO: ${DATA_REPO}"
echo "INIT_SAMPLES: ${INIT_SAMPLES}"
echo "BUILDINGS_FILE_PREFIX: ${BUILDINGS_FILE_PREFIX}"
echo "OBJ_DIR: ${OBJ_DIR}"
echo "PLY_N_DIR: ${PLY_N_DIR}"
echo "VIEWPOINTS_DIR: ${VIEWPOINTS_DIR}"
echo "RENDERINGS_DIR: ${RENDERINGS_DIR}"
echo "SAMPLE_POINTS_DIR: ${SAMPLE_POINTS_DIR}"
echo "PLY_DIR_PREFIX: ${PLY_DIR_PREFIX}"
echo "GROUPS_DIR: ${GROUPS_DIR}"
echo "WITH_GROUPS: ${WITH_GROUPS}"
echo "CUT_AT: ${CUT_AT}"
echo "RIDGE_VALLEY_DIR: ${RIDGE_VALLEY_DIR}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "PER_COMPONENT: ${PER_COMPONENT}"
echo "SAMPLES: ${SAMPLES}"
echo "EXPORT_CURVATURES: ${EXPORT_CURVATURES}"
echo "EXPORT_PCA: ${EXPORT_PCA}"
echo "OVERRIDE: ${OVERRIDE}"
echo "REMOVE: ${REMOVE}"
echo "PTS_ON_STYLE: ${PTS_ON_STYLE}"
echo "ON_GPU: ${ON_GPU}"
echo "RENDER_MODE: ${RENDER_MODE}"


run_args=""


function make_sub_file(){
  local input_file=$1
  local output_file=$2
  local start_idx=$3
  local end_idx=$4
  local current_idx=-1

  if [ -f "$output_file" ]; then
    echo "Remove existing $output_file"
    rm $output_file
  fi

  echo "Making subfile $output_file for job"
  while IFS= read -r line
  do
    current_idx=$((current_idx+1))
    if [ $current_idx -ge $((start_idx)) ] && [ $current_idx -lt $((end_idx)) ]; then
      echo "$line" >> $output_file
    fi
  done < "$input_file"

  return 1
}

run_jobs(){
  local job_id=-1
  local start_idx=${START_IDX}
  while [ $((start_idx)) -lt $((END_IDX)) ]; do

    job_id=$((job_id+1))
    job_name="${NAME}_J$job_id"
    local end_idx=$((start_idx + STEP_SIZE))
    if [ $end_idx -gt $END_IDX ]; then
      end_idx=$((END_IDX))
    fi

    local batch_suffix="_${NAME}_${start_idx}to${end_idx}"

    # NOTE: make sure to remove any previous buildings_....csv files otherwise if same name appears it won't replace it
    make_sub_file "$BUILDINGS_FILE_PREFIX.csv" \
                  "$BUILDINGS_FILE_PREFIX${batch_suffix}.csv" \
                   $start_idx \
                   $end_idx
    local needs_to_run=$?
    if [ $((needs_to_run)) -eq 1 ]; then
      $USE_FUNCTION
      date >> $LOGS_FILE
      if [ $((ON_CLUSTER)) -eq 1 ]; then
        sbatch_args="--job-name $job_name"
        sbatch_args="$sbatch_args $SBATCH_ARGS"
        echo "sbatch $sbatch_args ${JOB_FILE} ${run_args}" >> $LOGS_FILE
        sbatch $sbatch_args ${JOB_FILE} ${run_args}
      else
        echo "bash ${JOB_FILE} ${run_args}" >> $LOGS_FILE
        bash ${JOB_FILE} ${run_args}
      fi
    else
      echo "No need to run for $batch_suffix"
    fi

    start_idx=$((start_idx+STEP_SIZE))

  done
}


run_coarse_detailed_plys(){
  echo "in run_coarse_detailed_plys"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $PY_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $DATA_ROOT"
  run_args="${run_args} $DATA_REPO"
  run_args="${run_args} $SAMPLES_OUT_DIR/$PLY_DIR_PREFIX"
  run_args="${run_args} $DETAIL_SAMPLES"
  run_args="${run_args} $COARSE_SAMPLES"
}


run_pts2ply(){
  echo "in run_pts2ply"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $PY_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $INIT_SAMPLES"
  run_args="${run_args} $PLY_DIR_PREFIX"
  run_args="${run_args} $GROUPS_DIR"
  run_args="${run_args} $WITH_GROUPS"
  run_args="${run_args} $CUT_AT"
  run_args="${run_args} $RIDGE_VALLEY_DIR"
  run_args="${run_args} $NUM_PROCESSES"
  run_args="${run_args} $PER_COMPONENT"
  run_args="${run_args} $SAMPLE_POINTS_DIR"
  run_args="${run_args} $OBJ_DIR"
  run_args="${run_args} $COLOR_DIR"

  echo "args: $run_args"
}


run_meshsampling(){
  echo "in run_meshsampling"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $PY_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJ_DIR"
  run_args="${run_args} $NUM_PROCESSES"
  run_args="${run_args} $SAMPLES"
  run_args="${run_args} $EXPORT_CURVATURES"
  run_args="${run_args} $EXPORT_PCA"
  run_args="${run_args} $OVERRIDE"
  run_args="${run_args} $REMOVE"
  run_args="${run_args} $PTS_ON_STYLE"
  run_args="${run_args} $SAMPLE_POINTS_DIR"

  echo "args: $run_args"
}


run_retexture(){
  echo "in run_retexture"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $BLENDER_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJ_DIR"

  echo "args: $run_args"
}


run_uniquecomponents(){
  echo "in run_uniquecomponents"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $PY_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $DATA_ROOT/$DATA_REPO/$BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"
  run_args="${run_args} $SAMPLE_POINTS_DIR/$PLY_DIR_PREFIX"
  run_args="${run_args} $UNIQUE_DIR"
  run_args="${run_args} False"

  echo "args: $run_args"
}


run_viewpoints(){
  echo "in run_viewpoints"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $BLENDER_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJ_DIR"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"
  run_args="${run_args} $GROUPS_DIR"
  run_args="${run_args} $ON_GPU"
  run_args="${run_args} $RENDERINGS_DIR"
  run_args="${run_args} $VIEWPOINTS_DIR"
  run_args="${run_args} $UNIQUE_DIR"
  run_args="${run_args} $PLY_N_DIR"

  echo "args: $run_args"
}


run_renderings(){
  echo "in run_renderings"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $BLENDER_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJ_DIR"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"
  run_args="${run_args} $GROUPS_DIR"
  run_args="${run_args} $ON_GPU"
  run_args="${run_args} $RENDERINGS_DIR"
  run_args="${run_args} $VIEWPOINTS_DIR"
  run_args="${run_args} $UNIQUE_DIR"
  run_args="${run_args} $PLY_N_DIR"
  run_args="${run_args} $RENDER_MODE"

  echo "args: $run_args"
}


run_buildnetgrouping(){
  echo "in run_buildnetgrouping"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $BLENDER_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJ_DIR"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"
  run_args="${run_args} $GROUPS_DIR"
  run_args="${run_args} $ON_GPU"

  echo "args: $run_args"
}


run_unifyandnormalizecomponents(){
  echo "in run_unifyandnormalizecomponents"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $BLENDER_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJ_DIR"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"
  run_args="${run_args} $GROUPS_DIR"
  run_args="${run_args} $OBJCOMPONENTS_DIR"

  echo "args: $run_args"
}


run_rotatecomponents(){
  echo "in run_rotatecomponents"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $BLENDER_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJ_DIR"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"

  echo "args: $run_args"
}


run_addvolumetoplanes(){
  echo "in run_addvolumetoplanes"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $BLENDER_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJCOMPONENTS_DIR"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"
  run_args="${run_args} ${OBJCOMPONENTS_DIR}_volume"

  echo "args: $run_args"
}


run_buildnetrendercomponents(){
  echo "in run_buildnetrendercomponents"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $BLENDER_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"
  run_args="${run_args} $OBJCOMPONENTS_DIR"
  run_args="${run_args} $UNIQUE_DIR"

  echo "args: $run_args"
}


run_uploadgroups(){
  echo "in run_uploadgroups"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $PY_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $GOOGLE_DRIVE_REPO"
  run_args="${run_args} $DATA_ROOT/$DATA_REPO"
  run_args="${run_args} $GROUPS_DIR"

  echo "args: $run_args"
}


run_colourclouds(){
  echo "in run_colourclouds"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $PY_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $OBJ_DIR"
  run_args="${run_args} $SAMPLE_POINTS_DIR"
  run_args="${run_args} $INIT_SAMPLES"

  echo "args: $run_args"
}


run_pslabel(){
  echo "in run_pslabel"
  run_args=""
  run_args="${run_args} $job_name"
  run_args="${run_args} $PY_EXE"
  run_args="${run_args} $STYLE_REPO"

  run_args="${run_args} $BUILDINGS_FILE_PREFIX${batch_suffix}.csv"
  run_args="${run_args} $DATA_ROOT"
  run_args="${run_args} $DATA_REPO"
  run_args="${run_args} $INIT_SAMPLES"
  run_args="${run_args} $PLY_DIR_PREFIX"
  run_args="${run_args} $NUM_PROCESSES"
  run_args="${run_args} $SAMPLE_POINTS_DIR"
  run_args="${run_args} $OBJ_DIR"

  echo "args: $run_args"
}


run_jobs
