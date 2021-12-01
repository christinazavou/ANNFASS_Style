import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import STYLES, parse_components_with_style_csv


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--out_file", required=True, type=str)
parser.add_argument("--elements", type=str)
args = parser.parse_args()


elements = args.elements.split(",") if args.elements else None

os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

with open(args.out_file, "w") as f_test:
    for building_dir in os.listdir(args.data_dir):

        for component_file in os.listdir(os.path.join(args.data_dir, building_dir)):
            if elements and not any(e.lower() in component_file.lower() for e in elements):
                continue

            if not ".npy" in component_file:
                continue
            if "_labels.npy" in component_file:
                continue

            file = os.path.join(args.data_dir, building_dir, component_file)

            style = 'Unknown'
            labels = np.zeros((len(STYLES)))
            labels[0] = 1
            np.save(file.replace(".npy", "_labels.npy"), labels)

            if os.path.exists(file.replace(".npy", "_labels.npy")):
                line = "{};{}\n".format(file, file.replace(".npy", "_labels.npy"))
                f_test.write(line)
