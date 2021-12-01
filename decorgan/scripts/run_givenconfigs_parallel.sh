#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --nodes=1
#SBATCH --partition=GPU

export PYTHONUNBUFFERED="True"

SOURCE_DIR=${SOURCE_DIR:-/home/czavou01/decor-gan-private}
PY_EXE=${PY_EXE:-/home/czavou01/miniconda3/envs/decorgan/bin/python}
CONFIG_FILES=${CONFIG_FILES:-"ena.yml,dio.yml,tria.yml"}

echo "SOURCE_DIR: ${SOURCE_DIR}"
echo "PY_EXE: ${PY_EXE}"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

run_command_on_gpu(){
  config_file=$1
  out_file="${config_file%.yml}.out"
  err_file="${config_file%.yml}.err"
  export CONFIG_YML=$config_file \
    && export GPU=$2 \
    && sh ./run_mymain.sh > $out_file 2>$err_file
}

gpu=0
processes=()
IFS=',' read -ra ADDR <<< "$CONFIG_FILES"
for file in "${ADDR[@]}"; do
  run_command_on_gpu $file $gpu &
  current_process=$!
  processes+=("$current_process")
  gpu=$((gpu+1))
done


for pid in ${processes[*]}; do
  echo "pid: $pid"
  wait $pid
done
