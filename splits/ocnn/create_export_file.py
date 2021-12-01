import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.utils import parse_buildings_csv


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--txt_file", type=str, required=True)
parser.add_argument("--buildings_csv", type=str, required=True)
FLAGS = parser.parse_args()


txt_file = FLAGS.txt_file


buildings = parse_buildings_csv(FLAGS.buildings_csv)
found_buildings = []

os.makedirs(os.path.dirname(txt_file), exist_ok=True)

building_files = os.listdir(FLAGS.data_dir)
building_files = [b.replace(".ply", "") for b in building_files]
with open(txt_file, "w") as fout:
    for building_file in building_files:
        if building_file in buildings:
            found_buildings.append(building_file)
            fout.write(f"{building_file}.ply 50\n")

not_found_buildings = set(buildings) - set(found_buildings)
print(f"not_found_buildings: {not_found_buildings}")
