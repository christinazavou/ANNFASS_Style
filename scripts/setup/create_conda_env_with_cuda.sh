#!/bin/bash

source /home/graphicslab/miniconda3/etc/profile.d/conda.sh

env_name=$1
envs_path=$2
cuda_version=$3
py_version=$4

echo "------------------------------------------------"
echo "env_name:      ${env_name}"
echo "envs_path:     ${envs_path}"
echo "cuda_version:  ${cuda_version}"
echo "py_version:    ${py_version}"
echo "------------------------------------------------"

conda create --yes --name $env_name python=$py_version

scripts_path=$envs_path/$env_name/etc/conda

# Scripts under this folder are run whenever conda environment is activated
mkdir -p "$scripts_path/activate.d"
touch "$scripts_path/activate.d/activate.sh"

# replace argument one of file set_cuda_version.sh with this cuda_version
sed "s|\$1|${cuda_version}|g" set_cuda_version.sh >> "$scripts_path/activate.d/activate.sh"

# Scripts under this folder are run whenever conda environment is deactivated
mkdir -p "$scripts_path/deactivate.d"
cat unset_cuda_version.sh > "$scripts_path/deactivate.d/deactivate.sh"
