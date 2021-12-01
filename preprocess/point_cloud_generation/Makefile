#git clone https://github.com/christinazavou/mesh_sampling.git
SHELL:=/bin/bash # otherwise we can't use "source"


CONDA_ENV ?= style_detect_env
LOCAL_CONDA_PATH ?= /home/graphicslab/miniconda3
LOCAL_PY_EXE := ${LOCAL_CONDA_PATH}/envs/$(CONDA_ENV)/bin/python
SWARM2_PY_EXE := /home/maverkiou/miniconda3/envs/style_detect_env/bin/python
SFE_EXE := ../mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter
CLUSTER_ROOT_DIR := /mnt/nfs/work1/kalo/maverkiou/zavou/data
LOCAL_ROOT_DIR:=/media/graphicslab/BigData/zavou/ANNFASS_DATA

REPO ?= ANNFASS_Buildings_march
SAMPLES ?= 100000000
INIT_SAMPLES ?= 100000K
PLY_DIR_PREFIX ?= ply
CUT_AT ?= 1000000
JOB_NAME ?= PLYGENA
MAX_SAMPLES ?= 100K
BUILDING_FILE ?= buildings


pts2ply-swarm2:
	sbatch --job-name ${JOB_NAME} --nodes=1 ${SWARM2_PY_EXE} pts2ply.py \
			--root ${CLUSTER_ROOT_DIR} \
			--repo ${REPO} \
			--pts_dir samplePoints/point_cloud_${INIT_SAMPLES} \
			--ply_dir_prefix samplePoints/${PLY_DIR_PREFIX} \
			--cut_at ${CUT_AT} \
		> ${CLUSTER_ROOT_DIR}/${REPO}/samplePoints/pts_${INIT_SAMPLES}2ply_${CUT_AT}_False.log

pts2ply-local:
	${LOCAL_PY_EXE} pts2ply.py \
			--root ${LOCAL_ROOT_DIR} \
			--repo ${REPO} \
			--pts_dir samplePoints/point_cloud_${INIT_SAMPLES} \
			--ply_dir_prefix samplePoints/${PLY_DIR_PREFIX} \
			--cut_at ${CUT_AT} \
		> ${LOCAL_ROOT_DIR}/${REPO}/samplePoints/pts_${INIT_SAMPLES}2ply_${CUT_AT}_False.log

pts2ply-stylecomponent-local:
	${LOCAL_PY_EXE} pts2ply.py \
			--root $(LOCAL_ROOT_DIR) \
			--repo $(REPO) \
			--pts_dir samplePoints/point_cloud_$(INIT_SAMPLES)_style_mesh \
			--ply_dir_prefix samplePoints/$(PLY_DIR_PREFIX) \
			--buildings_csv $(BUILDING_FILE).csv \
			--cut_at $(CUT_AT) \
		> $(LOCAL_ROOT_DIR)/$(REPO)/samplePoints/pts_$(INIT_SAMPLES)2ply_${CUT_AT}_True.log

pts2ply-stylecomponent-local-one:
	${LOCAL_PY_EXE} pts2ply.py \
			--root /media/graphicslab/BigData/zavou/ANNFASS_DATA \
			--repo ANNFASS_Buildings_march \
			--pts_dir samplePoints/point_cloud_10000K_style_mesh \
			--ply_dir samplePoints/ply_max100K_per_component \
			--building 02_Panagia_Chrysaliniotissa \
			--cut_at 100000


point-cloud-debug:
	$(LOCAL_PY_EXE) create_point_cloud.py \
		--num_samples 100000 \
		--sfe $(SFE_EXE) \
		--export_curvatures False \
		--export_pca False \
		--override True \
		--remove True \
		--pts_dir /media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_manual/samplePoints/points_100K \
		--face_dir /media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_manual/samplePoints/faces_100K \
		--model_id 10_Axiothea_House \
		--model_filename /media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_manual/normalizedObj/10_Axiothea_House/10_Axiothea_House.obj

pts2ply-debug:
	$(LOCAL_PY_EXE) pts2ply.py \
			--root /home/christina/Documents/annfass_playground \
			--repo triangulated_meshes \
			--obj_dir obj \
			--building COMMERCIALfactory_mesh1454 \
			--pts_dir point_cloud_30K_c \
			--ply_dir ply_sampled_c \
			--cut_at 30000

