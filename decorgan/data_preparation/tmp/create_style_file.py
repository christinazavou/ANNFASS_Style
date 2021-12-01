import os
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/groups_june17_uni_nor_components_column_train.txt")
parser.add_argument("--out_file", type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/groups_june17_uni_nor_components_column_styles32.txt")
parser.add_argument("--num_styles", type=int, default=32)
FLAGS = parser.parse_args()


s = 0
with open(FLAGS.in_file, "r") as fin, \
        open(FLAGS.out_file, "w") as fout:
    for line in fin.readlines():
        rand = random.random()
        if s < FLAGS.num_styles and rand < 0.1:
            s+=1
            fout.write(line)
