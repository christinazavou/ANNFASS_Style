import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.utils import parse_buildings_csv


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--elements", type=str, )
parser.add_argument("--txt_file", type=str, required=True)
parser.add_argument("--buildings_csv", type=str, required=True)
FLAGS = parser.parse_args()


if FLAGS.elements:
    elements = FLAGS.elements.split(",")

txt_file = FLAGS.txt_file

buildings = parse_buildings_csv(FLAGS.buildings_csv)

found_buildings = []

os.makedirs(os.path.dirname(txt_file), exist_ok=True)

with open(txt_file, "w") as fout:
    for building in os.listdir(FLAGS.data_dir):
        if building in buildings:
            found_buildings.append(building)
            for component in os.listdir(os.path.join(FLAGS.data_dir, building)):
                if FLAGS.elements:
                    if not any(e.lower() in component.lower() for e in elements):
                        print(f"skipping {component}")
                        continue
                fout.write(f"{building}/{component}/model.obj\n")

not_found_buildings = set(buildings) - set(found_buildings)
print(f"not_found_buildings: {not_found_buildings}")
