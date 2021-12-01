import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--duplicates_csv", type=str, help="duplicates_csv", required=True)
parser.add_argument("--visualize_dir", type=str, help="visualize_dir", required=True)
parser.add_argument("--big_dir", type=str, help="big_dir", required=True)
parser.add_argument("--query_dir", type=str, help="query_dir", required=True)
ARGS = parser.parse_args()
print(ARGS)

os.makedirs(ARGS.visualize_dir, exist_ok=True)
with open(ARGS.duplicates_csv, "r") as fin:
    fin.readline()
    for line in fin.readlines():
        query, reference, _, _, _ = line.split()
        obj_query = os.path.join(ARGS.query_dir, query, "model.obj")
        if not os.path.exists(obj_query):
            query = query.lower()
            if query in ["06", "11", "19", "33"]:
                query = f"{query}_barcelona"
            obj_query = os.path.join(ARGS.query_dir, query, "model.obj")
            assert os.path.exists(obj_query)
        obj = os.path.join(ARGS.visualize_dir, f"{query}.obj")
        shutil.copyfile(obj_query, obj)
        obj_reference = os.path.join(ARGS.big_dir, reference, "model.obj")
        obj = os.path.join(ARGS.visualize_dir, f"{reference}.obj")
        shutil.copyfile(obj_reference, obj)
