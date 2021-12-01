from os import listdir, makedirs
from os.path import join
from distutils.dir_util import copy_tree
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--from_dir", type=str)
parser.add_argument("--to_dir", type=str)
parser.add_argument("--buildings_csv", type=str, default=None)
parser.add_argument("--depth", type=int, default=2)
FLAGS = parser.parse_args()

from_dir = FLAGS.from_dir
to_dir = FLAGS.to_dir
buildings_csv = FLAGS.buildings_csv
depth = FLAGS.depth


def parse_buildings_csv(filename):
    buildings = []
    with open(filename, "r") as f:
        for line in f:
            buildings.append(line.strip().split(";")[1])
    print("buildings to process: {}".format(buildings))
    return buildings


if buildings_csv is not None:
    include_buildings = parse_buildings_csv(buildings_csv)
else:
    include_buildings = 'all'

makedirs(to_dir, exist_ok=True)
for building_dir in listdir(from_dir):
    if include_buildings == 'all' or building_dir in include_buildings:
        if depth == 1:
            copy_tree(join(from_dir, building_dir), join(to_dir, building_dir))
        else:
            for component_dir in listdir(join(from_dir, building_dir)):
                copy_tree(join(from_dir, building_dir, component_dir), join(to_dir, building_dir, component_dir))
