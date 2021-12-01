import argparse
import os
import sys

from itertools import takewhile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import STYLES, parse_components_with_style_csv


parser = argparse.ArgumentParser()
parser.add_argument("--data_dirs", required=True, type=str)
parser.add_argument("--out_txt", required=True, type=str)
parser.add_argument("--components_csv", required=True, type=str)
args = parser.parse_args()


data_dirs = args.data_dirs.split(",")
components = parse_components_with_style_csv(args.components_csv)

os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)

common_root = ''.join(c[0] for c in takewhile(lambda x: all(x[0] == y for y in x), zip(*data_dirs)))

with open(args.out_txt, "w") as f_test:
    f_test.write(f"{common_root};;\n")
    for data_dir in data_dirs:
        for building_dir in os.listdir(data_dir):
            building_components = [(s, b, c) for (s, b, c) in components if building_dir == b]
            if len(building_components) == 0:
                continue
            building_components = [(s, b, c.replace("style_mesh_", "")) for (s, b, c) in building_components]

            for group_dir in os.listdir(os.path.join(data_dir, building_dir)):

                group = group_dir.replace("_", "")

                for component_view_file in os.listdir(os.path.join(data_dir, building_dir, group_dir)):
                    if not ".png" in component_view_file:
                        continue

                    file = os.path.join(data_dir, building_dir, group_dir, component_view_file)
                    component = group + "_" + "_".join(component_view_file.split("_")[:-3])

                    style = [s for (s, b, c) in building_components if c == component]
                    if len(style) == 0:
                        continue
                    assert len(style) == 1
                    style = style[0]

                    line = f"{data_dir}/{building_dir}/{group_dir}/{component_view_file};" \
                           f"{building_dir}/{component};" \
                           f"{STYLES.index(style)}\n"
                    f_test.write(line)
