import json
import os
import sys
from os.path import join, realpath, dirname

import numpy as np

SOURCE_DIR = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(join(SOURCE_DIR))

from preprocess.blender.io_utils import load_ply
from preprocess.blender.scene_utils import cleanup, bpy


# unique_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/newgroup_unique_point_clouds"
# target_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_component_refined"
group_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/group_triangles_to_component_v2"
ply_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedPly"


def run_for_building():
    cleanup(True)
    building = "RELIGIOUSchurch_mesh2460"
    building_ply = join(ply_dir, building, f"{building}.ply")
    load_ply(building_ply)
    building_obj = bpy.context.active_object
    building_dimensions = building_obj.dimensions
    dimensions = {building: f"{building_dimensions.x} {building_dimensions.y} {building_dimensions.z}"}
    # for component_file in os.listdir(join(unique_dir, building)):
    for group_bbox_file in os.listdir(join(group_dir, building)):
        # if not component_file.endswith(".ply"):
        if not group_bbox_file.endswith("obb.ply"):
            continue
        # component = component_file.replace("style_mesh_", "").split("_")[0]
        # _, group = component.split("group")[1]
        # group_obb_name = f"group_{group}_obb"
        group_obb_name = group_bbox_file.replace(".ply", "")
        # group_bbox_file = join(group_dir, building, f"group_{group}_obb.ply")
        # load_ply(group_bbox_file)
        load_ply(join(group_dir, building, group_bbox_file))
        bbox = bpy.data.objects[group_obb_name]
        group_dimensions = bbox.dimensions
        dimensions[group_obb_name] = f"{group_dimensions.x} {group_dimensions.y} {group_dimensions.z}"
        group_dim_pct = np.divide(np.array(group_dimensions), np.array(building_dimensions))
        dimensions[f"{group_obb_name}_pct"] = f"{group_dim_pct[0]} {group_dim_pct[1]} {group_dim_pct[2]}"
    # dim_file = join(target_dir, building, f"dimensions.json")
    dim_file = join(group_dir, building, f"dimensions.json")
    with open(dim_file, "w") as fout:
        json.dump(dimensions, fout, indent=4)


run_for_building()

# /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender --background --python get_dimensions.py
