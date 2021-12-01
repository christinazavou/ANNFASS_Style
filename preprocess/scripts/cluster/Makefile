SHELL := /usr/bin/bash # otherwise we can't use "source"

CONDA_ENV ?= style_detect_env
ANNFASS_REPO := ANNFASS_Buildings_march
BUILDNET_REPO := BUILDNET_Buildings
ROOT_DIR := /mnt/nfs/work1/kalo/maverkiou/zavou/data
CONDA_PATH := /home/maverkiou/miniconda3
#SWARM2_CONDA_PATH := /home/maverkiou/miniconda3
#GYPSUM_CONDA_PATH := /home/maverkiou/miniconda2
PY_EXE := ${CONDA_PATH}/envs/$(CONDA_ENV)/bin/python
BLENDER_EXE := /home/maverkiou/blender-2.91.0-linux64/blender
#BLENDER_EXE := /home/maverkiou/blender-2.93.5-linux-x64/blender
STYLE_REPO := /home/maverkiou/zavou/style_detection
SFE_EXE := ../../mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter

GOOGLE_DRIVE_REPO ?= BUILDNET_Buildings
PY_EXE ?= /home/maverkiou/miniconda2/envs/style_detect_env/bin/python
SAMPLES ?= 10000000
INIT_SAMPLES ?= 10000K
OBJ_DIR ?= normalizedObj
PLY_DIR_PREFIX ?= ply
SAMPLES_OUT_DIR ?= samplePoints
CUT_AT ?= 100000
PTS_ON_STYLE ?= False
NUM_PROCESSES ?= 8
START_IDX ?= 0
END_IDX ?= 30
BUILDING_FILE ?= buildings
RIDGE_VALLEY_DIR ?= "ridge_or_valley"
COLOR_DIR ?= "color"
STEP_SIZE ?= 400
REPO ?= BUILDNET_Buildings
MEMORY ?= 28GB
PARTITION ?= longq
PER_COMPONENT ?= False
WITH_GROUPS ?= False
GROUPS_DIR ?= groups
UNIQUE_DIR ?= unique_point_clouds
DETAIL_SAMPLES ?= 4096
COARSE_SAMPLES ?= 512
ON_GPU ?= True
WITH_GROUPS ?= True

retexture:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="RETEXTURE_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/retexture_job.sh" \
		&& export USE_FUNCTION=run_retexture \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& bash ../../schedule_same_jobs.sh

rotate-buildings:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="ROTATE_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/rotate_job.sh" \
		&& export USE_FUNCTION=run_rotatecomponents \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& bash ../../schedule_same_jobs.sh

buildnet-group-triangles-as-component:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="BCGROUPS_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/buildnet_grouping_job.sh" \
		&& export USE_FUNCTION=run_buildnetgrouping \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export ON_GPU=$(ON_GPU) \
		&& bash ../../schedule_same_jobs.sh

buildnet-unifynnormalize-components:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="BUNNNOR_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/buildnet_unify_and_normalize_job.sh" \
		&& export USE_FUNCTION=run_buildnetunifyandnormalizecomponents \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export OBJCOMPONENTS_DIR=$(OBJCOMPONENTS_DIR) \
		&& bash ../../schedule_same_jobs.sh

upload-groups:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="UPLOADGR_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/upload_groups_job.sh" \
		&& export USE_FUNCTION=run_uploadgroups \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export GOOGLE_DRIVE_REPO=$(GOOGLE_DRIVE_REPO) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& bash ../../schedule_same_jobs.sh

# NOTE: if you transfer obj files from local to cluster .. make sure mtl paths on cluster are correct (e.g. relative paths)
viewpoints:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="BVIEW_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/viewpoints_job.sh" \
		&& export USE_FUNCTION=run_viewpoints \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export UNIQUE_DIR=$(UNIQUE_DIR) \
		&& export VIEWPOINTS_DIR=$(VIEWPOINTS_DIR) \
		&& export RENDERINGS_DIR=$(RENDERINGS_DIR) \
		&& export PLY_N_DIR=$(PLY_N_DIR) \
		&& export ON_GPU=$(ON_GPU) \
		&& bash ../../schedule_same_jobs.sh

select-views:
	cd ../../ && \
	sbatch --partition $(PARTITION) "cd $(STYLE_REPO)/preprocess && $(PY_EXE) select_views.py --viewpoints_dir $(VIEWPOINTS_DIR) --buildings_csv $(BUILDING_FILE).csv --logs_dir $(ROOT_DIR)/$(REPO)/logs_select_views"

render-freestyle:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="BRENDF_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/renderings_job.sh" \
		&& export USE_FUNCTION=run_renderings \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export UNIQUE_DIR=$(UNIQUE_DIR) \
		&& export VIEWPOINTS_DIR=$(VIEWPOINTS_DIR) \
		&& export RENDERINGS_DIR=$(RENDERINGS_DIR) \
		&& export PLY_N_DIR=$(PLY_N_DIR) \
		&& export ON_GPU=$(ON_GPU) \
		&& export RENDER_MODE=1 \
		&& bash ../../schedule_same_jobs.sh

render-materials:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="BRENDM_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/renderings_job.sh" \
		&& export USE_FUNCTION=run_renderings \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export UNIQUE_DIR=$(UNIQUE_DIR) \
		&& export VIEWPOINTS_DIR=$(VIEWPOINTS_DIR) \
		&& export RENDERINGS_DIR=$(RENDERINGS_DIR) \
		&& export PLY_N_DIR=$(PLY_N_DIR) \
		&& export ON_GPU=$(ON_GPU) \
		&& export RENDER_MODE=2 \
		&& bash ../../schedule_same_jobs.sh

#--------------------------------------------------following is on swarm2 ----------------------------------------#

point-clouds:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="SFE_$(REPO)" \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/mesh_sampling_job.sh" \
		&& export USE_FUNCTION=run_meshsampling \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--mem $(MEMORY) --cpus-per-task=$(NUM_PROCESSES) --partition=$(PARTITION)" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export REMOVE=False \
		&& export OVERRIDE=False \
		&& export ON_STYLE=$(PTS_ON_STYLE) \
		&& export NUM_PROCESSES=$(NUM_PROCESSES) \
		&& export SAMPLE_POINTS_DIR=$(SAMPLES_OUT_DIR) \
		&& bash ../../schedule_same_jobs.sh


point-clouds-annfass-curvature:
	chmod 777 ${SFE_EXE} \
		&& mkdir -p annfassjobs \
		&& cd annfassjobs \
		&& bash ../../run_mesh_sampling_jobs.sh \
			--JOB_NAME A_SAMPLES \
			--START_IDX ${START_IDX} \
			--END_IDX ${END_IDX} \
			--NUM_SAMPLES ${SAMPLES} \
			--BUILDINGS_FILE_PREFIX ${BUILDING_FILE} \
			--DATA_REPO ${ANNFASS_REPO} \
			--NUM_PROCESSES ${NUM_PROCESSES} \
			--JOB_FILE ../../mesh_sampling_job.sh \
			--OVERRIDE True \
			--REMOVE True \
			--ON_STYLE ${PTS_ON_STYLE} \
			--EXPORT_CURVATURES \


ridge-valley:
	export PTS_PROCESSES=10 \
		&& export DATA_ROOT_DIR=$(ROOT_DIR) \
		&& export REPO=$(BUILDNET_REPO) \
		&& export STYLE_DIR=/home/maverkiou/zavou/style_detection \
		&& export INIT_SAMPLES=$(INIT_SAMPLES) \
		&& export PY_EXE=$(PY_EXE) \
		&& sbatch --cpus-per-task=10 --partition=m40-long ridge_valley.sh

colour-point-clouds:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="COLORCLOUD_$(REPO)" \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/colour_point_cloud_job.sh" \
		&& export USE_FUNCTION=run_colourclouds \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION)" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export SAMPLE_POINTS_DIR=$(SAMPLES_OUT_DIR) \
		&& export INIT_SAMPLES=$(INIT_SAMPLES) \
		&& bash ../../schedule_same_jobs.sh

ps-label:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="PSLABEL_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/ps_label_job.sh" \
		&& export USE_FUNCTION=run_pslabel \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION)" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export PLY_DIR_PREFIX=$(PLY_DIR_PREFIX) \
		&& export SAMPLE_POINTS_DIR=$(SAMPLES_OUT_DIR) \
		&& export INIT_SAMPLES=$(INIT_SAMPLES) \
		&& export NUM_PROCESSES=$(NUM_PROCESSES) \
		&& bash ../../schedule_same_jobs.sh

GROUPS_DIR ?= /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/groups_june17
PLY_DIR_PREFIX ?= /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/ply
BUILDING_FILE ?= /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/buildings
SAMPLES_OUT_DIR ?= /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/samplePoints
OBJ_DIR ?= /mnt/nfs/work1/kalo/maverkiou/zavou/data/BUILDNET_Buildings/normalizedObj
pts2ply:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="PTS2PLYS_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/pts2ply_job.sh" \
		&& export USE_FUNCTION=run_pts2ply \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--mem $(MEMORY) --cpus-per-task=$(NUM_PROCESSES) --partition=$(PARTITION)" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export INIT_SAMPLES=$(INIT_SAMPLES) \
		&& export PLY_DIR_PREFIX=$(PLY_DIR_PREFIX) \
		&& export SAMPLE_POINTS_DIR=$(SAMPLES_PTS_DIR) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export WITH_GROUPS=$(WITH_GROUPS) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export CUT_AT=$(CUT_AT) \
		&& export RIDGE_VALLEY_DIR=$(RIDGE_VALLEY_DIR) \
		&& export COLOR_DIR=$(COLOR_DIR) \
		&& export NUM_PROCESSES=$(NUM_PROCESSES) \
		&& export PER_COMPONENT=False \
		&& bash ../../schedule_same_jobs.sh

pts2ply_with_group:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="PTS2PLYS_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/pts2ply_job.sh" \
		&& export USE_FUNCTION=run_pts2ply \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--mem $(MEMORY) --cpus-per-task=$(NUM_PROCESSES) --partition=$(PARTITION)" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export INIT_SAMPLES=$(INIT_SAMPLES) \
		&& export PLY_DIR_PREFIX=$(PLY_DIR_PREFIX) \
		&& export SAMPLE_POINTS_DIR=$(SAMPLES_OUT_DIR) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export WITH_GROUPS=True \
		&& export CUT_AT=$(CUT_AT) \
		&& export RIDGE_VALLEY_DIR=$(RIDGE_VALLEY_DIR) \
		&& export COLOR_DIR=$(COLOR_DIR) \
		&& export NUM_PROCESSES=$(NUM_PROCESSES) \
		&& export PER_COMPONENT=True \
		&& bash ../../schedule_same_jobs.sh

find-unique-components:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="UNIQUE_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/unique_components_job.sh" \
		&& export USE_FUNCTION=run_uniquecomponents \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--mem $(MEMORY) --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export SAMPLE_POINTS_DIR=$(SAMPLES_OUT_DIR) \
		&& export PLY_DIR_PREFIX=$(PLY_DIR_PREFIX) \
		&& export UNIQUE_DIR=$(UNIQUE_DIR) \
		&& bash ../../schedule_same_jobs.sh

content_style:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="COARSE_DETAIL_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/coarse_and_detailed_plys_job.sh" \
		&& export USE_FUNCTION=run_coarse_detailed_plys \
		&& export PY_EXE=$(PY_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export SAMPLES_OUT_DIR=$(SAMPLES_OUT_DIR) \
		&& export PLY_DIR_PREFIX=$(PLY_DIR_PREFIX) \
		&& export DETAIL_SAMPLES=$(DETAIL_SAMPLES) \
		&& export COARSE_SAMPLES=$(COARSE_SAMPLES) \
	&& bash ../../schedule_same_jobs.sh

#sbatch --cpus-per-task 16 --partition longq --mem 32GB --nodes=1 sbatch_20.sh
#!/bin/bash
#/home/maverkiou/zavou/style_detection/preprocess/mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter --do-not-rescale-shape --export-point-samples --num-point-samples 100000000 --input-shape="/mnt/nfs/work1/kalo/maverkiou/zavou/data/ANNFASS_Buildings_march/normalizedObj/20_Pancyprian_Gymnasium_in_Nicosia/20_Pancyprian_Gymnasium_in_Nicosia_no_mtl_obj.txt"
