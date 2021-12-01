import argparse
import os
import random

import pandas as pd


def get_element_from_filename(file):
    return file.split("__")[0].split("_")[-1]


parser = argparse.ArgumentParser()
parser.add_argument("--data_dirs", type=str, required=True)
parser.add_argument("--elements", type=str, default="window,door,tower_steeple,column,railing_baluster,balcony_patio,dome,entrance_gate,parapet_merlon,buttress,dormer,arch")
parser.add_argument("--buildings_csv", type=str, required=True)
parser.add_argument("--txt_file", type=str, required=True)
parser.add_argument("--per_element", type=str, default="False")
FLAGS = parser.parse_args()


per_element = eval(FLAGS.per_element)
elements = FLAGS.elements.split(",")


txt_file = FLAGS.txt_file
buildings_df = pd.read_csv(FLAGS.buildings_csv, sep=';', header=None)
buildings_set_B_C = list(buildings_df[1].values)

os.makedirs(os.path.dirname(txt_file), exist_ok=True)


txt_file_A = txt_file.replace(".txt", "_setA.txt")
txt_file_BC = txt_file.replace(".txt", "_setBC.txt")

with open(txt_file_A, "w") as foutA:
    with open(txt_file_BC, "w") as foutBC:
        for data_dir in FLAGS.data_dirs.split(","):
            for building in os.listdir(data_dir):
                if not per_element:
                    if building in buildings_set_B_C:
                        foutBC.write(f"{building}\n")
                    else:
                        foutA.write(f"{building}\n")
                else:
                    for component in os.listdir(os.path.join(data_dir, building)):
                        if not any(e in component.lower() for e in elements):
                            print(f"skipping {component}")
                            continue
                        if building in buildings_set_B_C:
                            foutBC.write(f"{building}/{component}\n")
                        else:
                            foutA.write(f"{building}/{component}\n")

if per_element:
    txt_file_Abalance = txt_file.replace(".txt", "_setAbalanced.txt")

    lines_per_element = {}
    with open(txt_file_A, "r") as fin:
        for line in fin.readlines():
            el = get_element_from_filename(line.strip())
            lines_per_element.setdefault(el, [])
            lines_per_element[el] += [line]

    for element, lines in lines_per_element.items():
        lines_before = len(lines)
        random.shuffle(lines_per_element[element])
        lines_per_element[element] = lines_per_element[element][0:500]
        print(f"{element}, {lines_before}, {len(lines_per_element[element])}")

    with open(txt_file_A, "r") as fin, open(txt_file_Abalance, "w") as fout:
        for line in fin.readlines():
            element = get_element_from_filename(line)
            if line in lines_per_element[element]:
                fout.write(line)
