import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_buildings/normalizedObj")
parser.add_argument("--elements", type=str, default="window,door,tower_steeple,column,railing_baluster,balcony_patio,dome,entrance_gate,parapet_merlon,buttress,dormer,arch")
parser.add_argument("--buildings_csv", type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings.csv")
parser.add_argument("--txt_file", type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/buildnet_buildings.txt")
parser.add_argument("--per_element", type=str, default="False")
FLAGS = parser.parse_args()


def parse_buildings_csv(filename):
    buildings = []
    with open(filename, "r") as f:
        for line in f:
            buildings.append(line.strip().split(";")[1])
    print("buildings to process: {}".format(buildings))
    return buildings


per_element = eval(FLAGS.per_element)
elements = FLAGS.elements.split(",")
content_and_style_file = FLAGS.txt_file
buildings = parse_buildings_csv(FLAGS.buildings_csv)
train_buildings_cnt = int(len(buildings) * 0.8)
train_buildings = buildings[:train_buildings_cnt]

os.makedirs(os.path.dirname(content_and_style_file), exist_ok=True)

content_and_style_train_file = content_and_style_file.replace(".txt", "_train.txt")
content_and_style_val_file = content_and_style_file.replace(".txt", "_val.txt")

with open(content_and_style_train_file, "w") as fout_train, \
        open(content_and_style_val_file, "w") as fout_val:
    for building in os.listdir(FLAGS.data_dir):
        if building not in buildings:
            continue
        if not per_element:
            if building in train_buildings:
                fout_train.write(f"{building}\n")
            else:
                fout_val.write(f"{building}\n")
        else:
            for component in os.listdir(os.path.join(FLAGS.data_dir, building)):
                if not any(e in component.lower() for e in elements):
                    print(f"skipping {component}")
                    continue
                if building in train_buildings:
                    fout_train.write(f"{building}/{component}\n")
                else:
                    fout_val.write(f"{building}/{component}\n")
