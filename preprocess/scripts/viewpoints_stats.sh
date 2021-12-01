#!/bin/bash
#---------------------------------------------------------------------------------------------------------------------#

if [ -d /mnt/nfs/work1/kalo/maverkiou/zavou/data ]; then
  ROOT_DIR=/mnt/nfs/work1/kalo/maverkiou/zavou/data
  PY_FILE=/home/maverkiou/zavou/style_detection/preprocess
elif [ -d /media/christina/Data/ANNFASS_data ]; then
  ROOT_DIR=/media/christina/Data/ANNFASS_data
  PY_FILE=/media/christina/Data/ANNFASS_code/zavou-repos/style_detection/preprocess
elif [ -d /media/graphicslab/BigData/zavou/ANNFASS_DATA ]; then
  ROOT_DIR=/media/graphicslab/BigData/zavou/ANNFASS_DATA
  PY_FILE=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/preprocess
fi

DATA_REPO=$1
DATA_REPO=${DATA_REPO:="ANNFASS_Buildings"}
VIEWPOINTS_DIR=$2
VIEWPOINTS_DIR=${VIEWPOINTS_DIR:="viewpoints"}
VIEWPOINTS_DIR=${ROOT_DIR}/${DATA_REPO}/${VIEWPOINTS_DIR}
echo "ROOT_DIR: $ROOT_DIR"
echo "VIEWPOINTS_DIR: $VIEWPOINTS_DIR"

RESULT_CSV="${ROOT_DIR}/${DATA_REPO}/viewpoints_stats.csv"
if [ -f $RESULT_CSV ]; then
  rm -f $RESULT_CSV
fi
#---------------------------------------------------------------------------------------------------------------------#

line="building total_viewpoints total_selected_viewpoints groups_with_viewpoints groups_with_selected_viewpoints"
echo $line >> $RESULT_CSV

find "${VIEWPOINTS_DIR}" -maxdepth 1 -mindepth 1 -type d | while read -r building_dir; do

  building=$(basename -- $building_dir)
  echo "building $building"
  counts=0
  counts_selected=0
  groups_with_viewpoints=0
  groups_with_viewpoints_selected=0

  while read -r views_file; do
    left_square_bracket_counts=$(fgrep -o [ $views_file | wc -l)
    left_square_bracket_counts=$((((left_square_bracket_counts-1))/4))
    views_selected_file="${views_file/.json/_selected.json}"
    left_square_bracket_counts_selected=$(fgrep -o [ $views_selected_file | wc -l)
    left_square_bracket_counts_selected=$((((left_square_bracket_counts_selected-1))/4))
    counts=$((counts+left_square_bracket_counts))
    counts_selected=$((counts_selected+left_square_bracket_counts_selected))
    if [ "$left_square_bracket_counts" -gt "0" ]; then
      groups_with_viewpoints=$((groups_with_viewpoints+1))
    fi
    if [ "$left_square_bracket_counts_selected" -gt "0" ]; then
      groups_with_viewpoints_selected=$((groups_with_viewpoints_selected+1))
    fi
  done <<< $(find "${building_dir}" -maxdepth 1 -mindepth 1 -type f -name "*views.json")

  line="$building $counts $counts_selected $groups_with_viewpoints $groups_with_viewpoints_selected"
  echo $line >> $RESULT_CSV

done

