import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--txt_file", type=str, required=True)
parser.add_argument("--expected", type=int, required=True)
parser.add_argument("--filename", type=str, default="model_filled.binvox")
FLAGS = parser.parse_args()


data_dir = FLAGS.data_dir
txt_file = FLAGS.txt_file
os.makedirs(os.path.dirname(txt_file), exist_ok=True)

export_models = []
for idx, model_name in enumerate(os.listdir(data_dir)):
    if os.path.exists(os.path.join(data_dir, model_name, FLAGS.filename)):
        export_models += [model_name+"\n"]
assert idx + 1 == FLAGS.expected, f"{idx + 1} != {FLAGS.expected}"
export_models = sorted(export_models)


with open(txt_file, "w") as fout:
    fout.writelines(export_models)

