SHELL := /bin/bash # otherwise we can't use "source"

CONDA_ENV ?= style_detect_env
ANNFASS_REPO := ANNFASS_Buildings_may
BUILDNET_REPO := BUILDNET_Buildings
CONDA_PATH := /home/graphicslab/miniconda3
PY_EXE := ${CONDA_PATH}/envs/$(CONDA_ENV)/bin/python
ROOT_DIR := /media/graphicslab/BigData1/zavou/ANNFASS_DATA
SFE_EXE := ../../mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter
STYLE_REPO := /media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection
BLENDER_EXE ?= /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender

GOOGLE_DRIVE_REPO ?= BUILDNET_Buildings
REPO ?= BUILDNET_Buildings
RAW_DIR ?= raw_data
TRIANGLE_DIR ?= triangledObj
OBJ_DIR ?= normalizedObj
PLY_N_DIR ?= normalizedPly
GROUPS_DIR ?= groups
RENDERINGS_DIR ?= renderings
VIEWPOINTS_DIR ?= viewpoints
UNIQUE_DIR ?= unique_point_clouds
OBJCOMPONENTS_DIR ?= unified_normalized_components
# 0.01 SAMPLES <= generated points <= 1.01 SAMPLES ==> use SAMPLES:=generated points / 0.01
SAMPLES_OUT_DIR ?= samplePoints
SAMPLES ?= 10000000
INIT_SAMPLES ?= 10000K
PLY_DIR_PREFIX ?= ply
PER_COMPONENT ?= False
CUT_AT ?= -1
PTS_ON_STYLE ?= False
NUM_PROCESSES ?= 8
START_IDX ?= 0
END_IDX ?= 4
STEP_SIZE ?= 2
RIDGE_VALLEY_DIR ?= "ridge_or_valley"
COLOR_DIR ?= "color"
BUILDING_FILE ?= buildings
DETAIL_SAMPLES ?= 4096
COARSE_SAMPLES ?= 512
ON_GPU ?= False
WITH_GROUPS ?= True
#----------------------------------------------------------------------------------------------------------------------#
init:
	bash ./initial_building_csv.sh

#----------------------------------------------------------------------------------------------------------------------#
triangulate-annfass:
	cd ../../blender/annfass && \
		$(BLENDER_EXE) \
			-b -noaudio \
			--python triangulate.py \
				-- \
				-root_data $(ROOT_DIR)/$(ANNFASS_REPO) \
				-raw_data $(RAW_DIR) \
				-obj_out $(TRIANGLE_DIR) \
				-buildings_csv $(BUILDING_FILE).csv \
				-logs_dir $(ROOT_DIR)/$(ANNFASS_REPO)/logs
normalize-annfass:
	cd ../../ \
		&& $(PY_EXE) normalizeobj.py \
			--root_dir $(ROOT_DIR)/$(ANNFASS_REPO) \
			--obj_dir_in $(TRIANGLE_DIR) \
			--obj_dir_out $(OBJ_DIR) \
			--buildings_csv $(BUILDING_FILE).csv \
			--override False

groups-annfass:
	cd ../../blender/annfass && \
	$(BLENDER_EXE) \
	    --background \
	    --python dummy_group_as_component.py \
	    -- \
	     -obj_dir $(ROOT_DIR)/$(ANNFASS_REPO)/$(OBJ_DIR) \
	     -group_dir $(ROOT_DIR)/$(ANNFASS_REPO)/$(GROUPS_DIR) \
	     -buildings_csv $(ROOT_DIR)/$(ANNFASS_REPO)/$(BUILDING_FILE).csv \
	     -logs_dir $(ROOT_DIR)/$(ANNFASS_REPO)/groups_logs

annfass_style_stats:
	sh ../style_stats.sh $(ANNFASS_REPO)


normalize-buildnet:
	cd ../../ \
		&& python normalizeobj.py \
			--root_dir $(ROOT_DIR)/$(BUILDNET_REPO) \
			--obj_dir_in $(TRIANGLE_DIR) \
			--obj_dir_out $(OBJ_DIR) \
			--buildings_csv $(BUILDING_FILE).csv \
			--override False

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

unifynnormalize-components:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="BUNNNOR_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/unify_and_normalize_job.sh" \
		&& export USE_FUNCTION=run_unifyandnormalizecomponents \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export OBJCOMPONENTS_DIR=$(OBJCOMPONENTS_DIR) \
		&& bash ../../schedule_same_jobs.sh

add_volume_to_planes:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="ADDVOL_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/add_volume_to_planes_job.sh" \
		&& export USE_FUNCTION=run_addvolumetoplanes \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJCOMPONENTS_DIR=$(OBJCOMPONENTS_DIR) \
		&& bash ../../schedule_same_jobs.sh

buildnet-render-components:
	mkdir -p "$(REPO)jobs" \
		&& cd "$(REPO)jobs" \
		&& export NAME="BRENCOM_$(REPO)" \
		&& export DATA_ROOT=$(ROOT_DIR) \
		&& export START_IDX=$(START_IDX) \
		&& export END_IDX=$(END_IDX) \
		&& export STEP_SIZE=$(STEP_SIZE) \
		&& export JOB_FILE="../../jobs/render_components_job.sh" \
		&& export USE_FUNCTION=run_buildnetrendercomponents \
		&& export BLENDER_EXE=$(BLENDER_EXE) \
		&& export STYLE_REPO=$(STYLE_REPO) \
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export GROUPS_DIR=$(GROUPS_DIR) \
		&& export OBJCOMPONENTS_DIR=$(OBJCOMPONENTS_DIR) \
		&& export UNIQUE_DIR=$(UNIQUE_DIR) \
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

#GROUPS_DIR ?= $(ROOT_DIR)/$(REPO)/groups_june17
#BUILDING_FILE ?= $(ROOT_DIR)/$(REPO)/buildings_torender
#OBJ_DIR ?= $(ROOT_DIR)/$(REPO)/normalizedObj
#UNIQUE_DIR ?= $(ROOT_DIR)/$(REPO)/groups_june17_unique_point_clouds
#VIEWPOINTS_DIR ?= $(ROOT_DIR)/$(REPO)/groups_june17_viewpoints
#RENDERINGS_DIR ?= $(ROOT_DIR)/$(REPO)/groups_june17_renderings
#PLY_N_DIR ?= $(ROOT_DIR)/$(REPO)/normalizedPly

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
	$(PY_EXE) select_views.py \
	    --viewpoints_dir $(VIEWPOINTS_DIR) \
	    --buildings_csv $(BUILDING_FILE).csv \
	    --logs_dir "$(ROOT_DIR)/$(REPO)/logs_select_views"

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

point-clouds-curvature:
	chmod 777 ${SFE_EXE} \
	&& bash ../run_mesh_sampling_jobs.sh \
		--JOB_NAME A_SAMPLES \
		--START_IDX ${START_IDX} \
		--END_IDX ${END_IDX} \
		--NUM_SAMPLES ${SAMPLES} \
		--BUILDINGS_FILE_PREFIX ${BUILDING_FILE} \
		--DATA_REPO ${ANNFASS_REPO} \
		--EXPORT_CURVATURES \
		--NUM_PROCESSES ${NUM_PROCESSES} \
		--JOB_FILE ../mesh_sampling_job.sh \
		--OVERRIDE True \
		--REMOVE True \
		--ON_STYLE ${PTS_ON_STYLE}

ridge-valley:
	cd ../../point_cloud_generation \
		&& ${PY_EXE} create_ridge_valley_clouds.py \
			--num_processes ${NUM_PROCESSES} \
			--buildings_csv ${ROOT_DIR}/${ANNFASS_REPO}/${BUILDING_FILE}.csv \
			--rnv /media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/preprocess/rtsc_utils/rtsc-1.6/build/RNV_Exporter_App \
			--obj_dir ${ROOT_DIR}/${ANNFASS_REPO}/normalizedObj \
			--pts_dir ${ROOT_DIR}/${ANNFASS_REPO}/samplePoints/point_cloud_${INIT_SAMPLES} \
			--override False \
			--remove False \
			--debug True

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
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
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
		&& export SBATCH_ARGS="--cpus-per-task 8 --partition $(PARTITION) --gres=gpu:1" \
		&& export BUILDINGS_FILE_PREFIX=$(BUILDING_FILE) \
		&& export DATA_REPO=$(REPO) \
		&& export OBJ_DIR=$(OBJ_DIR) \
		&& export PLY_DIR_PREFIX=$(PLY_DIR_PREFIX) \
		&& export SAMPLE_POINTS_DIR=$(SAMPLES_OUT_DIR) \
		&& export INIT_SAMPLES=$(INIT_SAMPLES) \
		&& export NUM_PROCESSES=$(NUM_PROCESSES) \
		&& bash ../../schedule_same_jobs.sh


GROUPS_DIR ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17
PLY_DIR_PREFIX ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/ply
BUILDING_FILE ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings
SAMPLES_PTS_DIR ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints
OBJ_DIR ?= /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj
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
		&& export SBATCH_ARGS="--partition $(PARTITION) --gres=gpu:1" \
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
		&& export SBATCH_ARGS="" \
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
		&& export SBATCH_ARGS="" \
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

