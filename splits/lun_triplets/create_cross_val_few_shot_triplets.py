import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="furniture", type=str)
parser.add_argument("--splits", default="splits3", type=str)
args = parser.parse_args()

ROOT_DIR = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/compressed_files/Data-all/Data"
SPLITS_DIR = f"{ROOT_DIR}/{args.data}/response/{args.splits}"

for root, dirs, files in os.walk(SPLITS_DIR):
    for file in files:
        if file.startswith("train"):
            file_in = os.path.join(root, file)
            for pct in [0.1, 0.3, 0.5]:
                file_out = file_in.replace(".txt", f"_{int(pct*10)}%.txt")
                with open(file_in, "r") as fin:
                    lines = fin.readlines()
                    random.shuffle(lines)
                    keep_size = int(len(lines) * pct)
                    lines = lines[:keep_size]
                with open(file_out, "w") as fout:
                    fout.writelines(lines)
