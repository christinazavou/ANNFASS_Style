#!/bin/bash
#SBATCH -o 01uniq.out
#SBATCH -e 01uniq.err
#SBATCH --time=5-00:00
#SBATCH --partition=longq
#SBATCH --cpus-per-task=8

datadir=/mnt/nfs/work1/kalo/maverkiou/zavou/data/ANNFASS_Buildings_march
blenderexe=/home/maverkiou/zavou/blender-2.91.0-linux64/blender
preprocessdir=/home/maverkiou/zavou/style_detection/preprocess

building=01_Cathedral_of_Holy_Wisdom

cd ${preprocessdir}/blender && ${blenderexe} --background --python unique_components.py -- -obj_file ${datadir}/normalizedObj/${building}/${building}.obj -out_dir ${datadir}/unique_9mar/${building} -style_elements_json ${preprocessdir}/resources/ANNFASS_STYLISTIC_ELEMENTS.json -chamfer_factor 0.01

