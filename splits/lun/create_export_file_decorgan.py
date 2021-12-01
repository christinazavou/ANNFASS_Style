import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--txt_file", type=str, required=True)
FLAGS = parser.parse_args()


data_dir = FLAGS.data_dir
txt_file = FLAGS.txt_file
os.makedirs(os.path.dirname(txt_file), exist_ok=True)

with open(txt_file, "w") as fout:
    for building in os.listdir(data_dir):
        for component in os.listdir(os.path.join(data_dir, building)):
            if os.path.exists(os.path.join(data_dir, building, component, "model_filled.binvox")):
                fout.write(f"{building}/{component}\n")
