import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/groups_june17_uni_nor_components_tower_steepledomedoorwindowcolumn_train.txt")
FLAGS = parser.parse_args()


def get_element_from_filename(file):
    return file.split("__")[0].split("_")[-1]


lines_per_element = {}
with open(FLAGS.in_file, "r") as fin:
    for line in fin.readlines():
        el = get_element_from_filename(line.strip())
        lines_per_element.setdefault(el, [])
        lines_per_element[el] += [line]


for element, lines in lines_per_element.items():
    print(f"{element}, {len(lines)}")
    random.shuffle(lines_per_element[element])
    lines_per_element[element] = lines_per_element[element][0:500]
    print(f"{element}, {len(lines)}")


with open(FLAGS.in_file, "r") as fin, open(FLAGS.in_file.replace(".txt", "balanced.txt"), "w") as fout:
    for line in fin.readlines():
        element = get_element_from_filename(line)
        if line in lines_per_element[element]:
            fout.write(line)

