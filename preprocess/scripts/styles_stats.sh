#!/bin/bash
#---------------------------------------------------------------------------------------------------------------------#

ROOT_DIR=${ROOT_DIR:-/media/graphicslab/BigData/zavou/ANNFASS_DATA}
PREPROCESS_DIR=${PREPROCESS_DIR:-/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/preprocess}
OBJ_DIR=${OBJ_DIR:-normalizedObj}
GROUPS_DIR=${GROUPS_DIR:-groups}
STYLES_FILE=${STYLES_FILE:-"../../resources/STYLES.txt"}

DATA_REPO=$1
echo "ROOT_DIR: $ROOT_DIR"
echo "OBJ_DIR: $OBJ_DIR"
echo "GROUPS_DIR: $GROUPS_DIR"
echo "STYLES_FILE: $STYLES_FILE"

RESULT_CSV="${ROOT_DIR}/${DATA_REPO}/style_stats.csv"
if [ -f $RESULT_CSV ]; then
  rm -f $RESULT_CSV
fi

IFS=$'\r\n' GLOBIGNORE='*' command eval 'classes=($(cat $STYLES_FILE))'
classes_length=${#classes[@]}
echo "We have $classes_length classes"
#---------------------------------------------------------------------------------------------------------------------#

line="building "
line+="${classes[@]}"
line+=" "
line+="${classes[@]}"
echo $line >> $RESULT_CSV

# todo: and not "_style_mesh.obj" in name
find "${ROOT_DIR}/${DATA_REPO}/${OBJ_DIR}" -maxdepth 1 -type d | while read -r building_dir; do

  building=${building_dir#"${ROOT_DIR}/${DATA_REPO}/${OBJ_DIR}/"}

  echo "building $building"

  obj_file="${ROOT_DIR}/${DATA_REPO}/${OBJ_DIR}/${building}/${building}.obj"

  class_frequencies_per_element=()
  search_string="__"
  parts=()

  for i in "${classes[@]}"; do class_frequencies_per_element+=(0); done

  components_with_style=($(cat "${obj_file}" | grep "$search_string"))

  for component in "${components_with_style[@]}"; do
    component_lower=$(echo "$component" | awk '{print tolower($0)}')

    for ((i=0; i<$classes_length; i++)); do
      class_lower=$(echo "${classes[i]}" | awk '{print tolower($0)}')

      if [[ $component_lower == *"$class_lower"* ]]; then

        part=${component_lower%$search_string*}
        part=$(printf '%s' "$part" | sed 's/[0-9]//g')
        if [[ ! " ${parts[*]} " == *" ${part} "* ]]; then
          parts+=("${part}")
        fi

        (( class_frequencies_per_element[$i]++ ))

      fi
    done

  done

  echo "${classes[@]}"
  echo "components" "${parts[@]}"  # to find out what elements have style and put them in ANNFASS_STYLISTIC_EMEMENTS.json

  class_frequencies_per_group=$(cd ${PREPROCESS_DIR} && python style_stats.py \
    --root_dir ${ROOT_DIR} \
    --data_repo ${DATA_REPO} \
    --groups_dir ${GROUPS_DIR} \
    --building $building)

  line="$building "
  line+="${class_frequencies_per_element[@]}"
  line+=" "
  line+="$class_frequencies_per_group"
  echo $line >> $RESULT_CSV


done