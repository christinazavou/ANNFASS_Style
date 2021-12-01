#!/bin/bash
#SBATCH -J smi
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=150000

SOURCE_DIR=${SOURCE_DIR:-/home/czavou01/decor-gan-private}
PY_EXE=${PY_EXE:-/home/czavou01/miniconda3/envs/decorgan/bin/python}
CONFIGS_DIR=${CONFIGS_DIR:-/home/czavou01/decor-gan-private/settings/turing1}
CONFIG=${CONFIG:-finetune/chair/adain_p2_in16_out128_g32d32}
GPU=${GPU:-0}
MAIN_FILE=${MAIN_FILE:-mymain}

export CUDA_VISIBLE_DEVICES=${GPU}

echo "start ${PY_EXE} ${SOURCE_DIR}/${MAIN_FILE}.py --config_yml ${CONFIGS_DIR}/${CONFIG}.yml --gpu ${GPU}"
${PY_EXE} ${SOURCE_DIR}/${MAIN_FILE}.py --config_yml ${CONFIGS_DIR}/${CONFIG}.yml --gpu ${GPU}
