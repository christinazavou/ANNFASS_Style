#!/bin/bash
#SBATCH -J smi
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=150000

SOURCE_DIR=${SOURCE_DIR:-/home/czavou01/3d-AAE}
PY_EXE=${PY_EXE:-/home/czavou01/miniconda3/envs/decorgan/bin/python}
CONFIG=${CONFIG:-buildnet/aae/turing/hyperparams.json}
GPU=${GPU:-0}
MAIN_FILE=${MAIN_FILE:-train_aae.py}

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=$PYTHONPATH:${SOURCE_DIR}:${SOURCE_DIR}/experiments
cd ${SOURCE_DIR}/experiments
args="--config ${SOURCE_DIR}/settings/${CONFIG}"
echo "start ${PY_EXE} ${MAIN_FILE} $args"
${PY_EXE} ${MAIN_FILE} $args
