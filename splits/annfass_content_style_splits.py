import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, type=str)
parser.add_argument("--logs_dir", required=True, type=str)
parser.add_argument("--repo", default="ANNFASS_Buildings_may", type=str)
parser.add_argument("--content_dir", default="samplePoints/stylePly_cut10.0K_pgc_content512", type=str)
parser.add_argument("--style_dir", default="samplePoints/stylePly_cut10.0K_pgc_style4096", type=str)
parser.add_argument("--splits_dir", default="annfass_content_style_splits_may", type=str)
parser.add_argument("--splits_json", default="annfass_splits_may/classification_cross_val.json", type=str)
args = parser.parse_args()


with open(os.path.join(args.logs_dir, args.splits_json), "r") as fin:
    cross_val_splits = json.load(fin)


ply_content_dir = f"{args.root_dir}/{args.repo}/{args.content_dir}"
ply_style_dir = f"{args.root_dir}/{args.repo}/{args.style_dir}"


for split in cross_val_splits:
    content_dir = os.path.join(args.logs_dir, args.splits_dir, f"fold{split}", "content")
    style_dir = os.path.join(args.logs_dir, args.splits_dir, f"fold{split}", "style")
    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)

    content_train_file = os.path.join(content_dir, "train.txt")
    content_test_file = os.path.join(content_dir, "test.txt")
    style_train_file = os.path.join(style_dir, "train.txt")
    style_test_file = os.path.join(style_dir, "test.txt")

    log_file = os.path.join(args.logs_dir, args.splits_dir, "no_components_found.txt")

    lf = open(log_file, "w")

    with open(content_train_file, "w") as f_content_train, \
        open(content_test_file, "w") as f_content_test,\
        open(style_train_file, "w") as f_style_train,\
        open(style_test_file, "w") as f_style_test:

        for building in cross_val_splits[split]["train_buildings"]:
            has_components = False
            if os.path.exists(os.path.join(ply_content_dir, building)) and os.path.exists(os.path.join(ply_style_dir, building)):
                for component_file in os.listdir(os.path.join(ply_content_dir, building)):
                    if component_file in os.listdir(os.path.join(ply_style_dir, building)):
                        has_components = True
                        assert component_file.endswith(".ply")
                        component_name = component_file.replace(building+"_style_mesh_", "")
                        filepath = os.path.join(ply_content_dir, building, component_file)
                        newline = "{};{};{}\n".format(filepath, building, component_name)
                        f_content_train.write(newline)
                        filepath = os.path.join(ply_style_dir, building, component_file)
                        newline = "{};{};{}\n".format(filepath, building, component_name)
                        f_style_train.write(newline)
            if not has_components:
                lf.write(building+"\n")
        for building in cross_val_splits[split]["test_buildings"]:
            has_components = False
            if os.path.exists(os.path.join(ply_content_dir, building)) and os.path.exists(os.path.join(ply_style_dir, building)):
                for component_file in os.listdir(os.path.join(ply_content_dir, building)):
                    if component_file in os.listdir(os.path.join(ply_style_dir, building)):
                        has_components = True
                        assert component_file.endswith(".ply")
                        component_name = component_file.replace(building+"_style_mesh_", "")
                        filepath = os.path.join(ply_content_dir, building, component_file)
                        newline = "{};{};{}\n".format(filepath, building, component_name)
                        f_content_test.write(newline)
                        filepath = os.path.join(ply_style_dir, building, component_file)
                        newline = "{};{};{}\n".format(filepath, building, component_name)
                        f_style_test.write(newline)
            if not has_components:
                lf.write(building+"\n")
    lf.close()
