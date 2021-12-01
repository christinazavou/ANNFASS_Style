import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="furniture", type=str)
args = parser.parse_args()

out_txt = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/export_{args.data}.txt"
data_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/{args.data}"
with open(out_txt, "w") as fout:
    for dir1 in os.listdir(data_dir):
        for dir2 in os.listdir(os.path.join(data_dir, dir1)):
            fout.write(f"{dir1}/{dir2}\n")


out_txt = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/{args.data}_styles32.txt"
data_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/{args.data}"
possible_lines = []
for dir1 in os.listdir(data_dir):
    dirs = os.listdir(os.path.join(data_dir, dir1))
    random.shuffle(dirs)
    possible_lines += [f"{dir1}/{dirs[0]}\n"]
random.shuffle(possible_lines)
selected_lines = possible_lines[:32]
with open(out_txt, "w") as fout:
    fout.writelines(selected_lines)

