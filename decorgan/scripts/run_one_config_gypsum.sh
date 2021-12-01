#!/bin/bash
#SBATCH -J smi
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=rtx8000-long
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=150000

SOURCE_DIR=${SOURCE_DIR:-/home/maverkiou/zavou/decor-gan-private}
PY_EXE=${PY_EXE:-/home/maverkiou/miniconda2/envs/decorgan/bin/python}
CONFIG_YML=${CONFIG_YML:-/home/maverkiou/zavou/decor-gan-private/settings/turing1/finetune/chair/adain_p2_in16_out128_g32d32.yml}

cd ${SOURCE_DIR}
echo "start ${PY_EXE} mymain.py --config_yml ${CONFIG_YML}"
${PY_EXE} mymain.py --config_yml ${CONFIG_YML}
