#!/bin/sh

if [ -d /media/christina/Data/ANNFASS_data ]; then
  ROOT_DIR=/media/christina/Data/ANNFASS_data
elif [ -d /media/graphicslab/BigData/zavou/ANNFASS_DATA ]; then
  ROOT_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA
else
  echo "Unknown location"
  exit 1
fi

LABELS_DIR=${ROOT_DIR}/BUILDNET_Buildings/raw_data/component_to_labels/GNN/label_32
BUILDNET_CSV=${ROOT_DIR}/BUILDNET_Buildings/buildings.csv
ANNFASS_DIR=${ROOT_DIR}/ANNFASS_Buildings/raw_data
ANNFASS_CSV=${ROOT_DIR}/ANNFASS_Buildings/buildings.csv

generate_buildnet_csv(){

  echo "Generating buildnet csv"

  labels_dir=$1
  buildings_csv=$2

  rm -f $buildings_csv

  find "${labels_dir}" -maxdepth 1 -mindepth 1 -type f -name "*" | while read -r building_file; do
    read -r filename <<< $(echo "$building_file" | awk 'BEGIN { FS = "/" } ; { print $(NF) }')
    building=$(echo ${filename//"_label.json"/""})
    line="Unknown Style;$building"
    echo "$line" >> "${buildings_csv}"
  done

}


generate_annfass_csv(){

  echo "Generating annfass csv"

  styles_dir=$1
  buildings_csv=$2

  rm -f $buildings_csv

  find "${styles_dir}" -maxdepth 2 -mindepth 2 -type d -name "*" | while read -r building_dir; do
    read -r building <<< $(echo "$building_dir" | awk 'BEGIN { FS = "/" } ; { print $(NF) }')
    read -r style <<< $(echo "$building_dir" | awk 'BEGIN { FS = "/" } ; { print $(NF-1) }')
    line="$style;$building"
    echo "$line" >> "${buildings_csv}"
  done

}

generate_buildnet_csv $LABELS_DIR $BUILDNET_CSV
generate_annfass_csv $ANNFASS_DIR $ANNFASS_CSV
