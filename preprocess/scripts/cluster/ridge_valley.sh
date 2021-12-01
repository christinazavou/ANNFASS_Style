#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=longq
#SBATCH --time=7-00:00

PTS_PROCESSES=${PTS_PROCESSES:-10}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-"/mnt/nfs/work1/kalo/maverkiou/zavou/data"}
REPO=${REPO:-"BUILDNET_Buildings"}
STYLE_DIR=${STYLE_DIR:-/home/maverkiou/zavou/style_detection}
INIT_SAMPLES=${INIT_SAMPLES:-}
PY_EXE=${PY_EXE:-/home/maverkiou/miniconda2/envs/style_detect_env/bin/python}

cd ../../point_cloud_generation \
		&& ${PY_EXE} create_ridge_valley_clouds.py \
			--num_processes ${PTS_PROCESSES} \
			--buildings_csv ${DATA_ROOT_DIR}/${REPO}/buildings.csv \
			--rnv ${STYLE_DIR}/preprocess/rtsc_utils/rtsc-1.6/build/RNV_Exporter_App \
			--obj_dir ${DATA_ROOT_DIR}/${REPO}/normalizedObj \
			--pts_dir ${DATA_ROOT_DIR}/${REPO}/samplePoints/point_cloud_${INIT_SAMPLES} \
			--override False \
			--remove False \
			--debug True
