import json
import os


def get_unique_dir_per_building(unique_dirs):
    unique_dir_per_building = {}
    for unique_dir in unique_dirs.split(","):
        buildings = list(os.listdir(os.path.join(root_dir, unique_dir)))
        for building in buildings:
            unique_dir_per_building[building] = unique_dir
    return unique_dir_per_building


def run(root_dir, unique_dir_per_building, inp_file, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(inp_file, "r") as fin:
        lines = fin.readlines()
    with open(out_file, "w") as fout:
        for line in lines:
            building = line.split("/")[-2]
            component = line.split("/")[-1]
            if building in unique_dir_per_building:
                unique_dir = unique_dir_per_building[building]
                duplicates_file = os.path.join(root_dir, unique_dir, building, "duplicates.json")
                with open(duplicates_file, "r") as fin:
                    duplicates = json.load(fin)
                uniques = [u.replace("style_mesh_", "").split(".")[0] for u in duplicates.keys()]
                if any(u in line for u in uniques):
                    fout.write(line)


if __name__ == '__main__':
    inp_dir = '/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/svm_mink_ps/data/buildnetply100Knocolor/combined_splits_final_unique/layer_n-2_features_weighted_sum_per_component_max/classification_cross_val'
    out_dir = '/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/svm_mink_ps/data/buildnetply100Knocolor/combined_splits_final_unique_/layer_n-2_features_weighted_sum_per_component_max/classification_cross_val'
    root_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/"
    unique_dirs = 'BUILDNET_Buildings/groups_june17_unique_point_clouds,ANNFASS_Buildings_may/unique_point_clouds'

    udpb = get_unique_dir_per_building(unique_dirs)
    for inp_file in os.listdir(inp_dir):
        run(root_dir, udpb, os.path.join(inp_dir, inp_file), os.path.join(out_dir, inp_file))

