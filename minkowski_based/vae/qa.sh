#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=1080ti-short    # Partition to submit to
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


export PYTHONUNBUFFERED="True"

PY_EXE=/home/maverkiou/miniconda2/envs/py3-mink/bin/python
SOURCE_DIR=/home/maverkiou/zavou/style_detection/minkoski_pytorch

cd ${SOURCE_DIR}
export PYTHONPATH=$(pwd):$(pwd)/examples
cd examples && ${PY_EXE} test_vae.py TestModelNet40Dataset.test_cluster

