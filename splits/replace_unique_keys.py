import os
import json
import pandas as pd


existing = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/common_ocnn_decor_minkps.txt"
include = pd.read_csv(existing, sep="/", header=None)
include = [(row[0], row[1]) for i, row in include.iterrows()]


unique_dirs = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds,/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds"

for unique_dir in unique_dirs.split(","):
    for building in os.listdir(unique_dir):
        duplicates_file = os.path.join(unique_dir, building, "duplicates.json")
        with open(duplicates_file, "r") as fin:
            duplicates = json.load(fin)
            new_duplicates = {}
            for unique, duplicate_components in duplicates.items():
                unique_component_group = int(unique.replace("style_mesh_", "").split("_")[0].replace("group", ""))
                if (building, unique_component_group) not in include:
                    found = False
                    for idx, component in enumerate(duplicate_components):
                        component_group = int(component.replace("style_mesh_", "").split("_")[0].replace("group", ""))
                        if (building, unique_component_group) in include:
                            found = True
                            break
                    if found:
                        del duplicate_components[idx]
                        duplicate_components.append(unique)
                        new_duplicates[component] = duplicate_components
                else:
                    new_duplicates[unique] = duplicate_components
        new_duplicates_file = duplicates_file.replace("unique_point_clouds", "unique_point_clouds_include")
        os.makedirs(os.path.dirname(new_duplicates_file), exist_ok=True)
        with open(new_duplicates_file, "w") as fout:
            json.dump(new_duplicates, fout, indent=2)
