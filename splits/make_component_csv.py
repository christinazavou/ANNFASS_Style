import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import STYLES, parse_buildings_with_style_csv

parser = argparse.ArgumentParser()
parser.add_argument("--component_dirs", required=True, type=str)
parser.add_argument("--unique_dirs", required=True, type=str)
parser.add_argument("--buildings_csv", required=True, type=str)
parser.add_argument("--components_csv", required=True, type=str)
parser.add_argument("--override_labels", required=True, type=str)
parser.add_argument("--parts", type=str, default=None)  # e.g. --parts "window,door,dome,column,tower"
args = parser.parse_args()


buildings_with_style = parse_buildings_with_style_csv(args.buildings_csv)

parts = args.parts.split(",") if args.parts is not None else None

with open(args.components_csv, "w") as fout:
    for data_dir, unique_dir, override_labels in zip(args.component_dirs.split(","),
                                                     args.unique_dirs.split(","),
                                                     args.override_labels.split(",")):
        for building_dir in os.listdir(data_dir):
            if building_dir not in os.listdir(unique_dir):
                continue
            for component_dir in os.listdir(os.path.join(data_dir, building_dir)):

                component = component_dir.replace("style_mesh_", "")
                unique_components = os.listdir(os.path.join(unique_dir, building_dir))
                unique_components = [os.path.splitext(u)[0] for u in unique_components]
                unique_components = [u.replace("style_mesh_", "") for u in unique_components]

                if component not in unique_components:
                    continue

                building_style = [s for (s, b) in buildings_with_style if building_dir in b]
                if len(building_style) == 0:
                    continue
                assert len(building_style) == 1
                building_style = building_style[0]

                if parts is not None:
                    if not any(p.lower() in component_dir.lower() for p in parts):
                        continue  # do not include this part

                component_style = [s for s in STYLES if s.lower() in component_dir.lower()]
                assert len(component_style) <= 1
                if len(component_style) == 1:
                    component_style = component_style[0]
                if len(component_style) == 0 or component_style.lower() == 'unknown' and override_labels:
                    component_style = building_style
                fout.write(f"{component_style};{building_dir};{component_dir}\n")
