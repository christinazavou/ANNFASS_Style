import argparse
import os

import numpy as np

from common.utils import UndirectedGraph

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="furniture", type=str)
args = parser.parse_args()

ROOT_DIR = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/compressed_files/Data-all/Data"
RESPONSE_DIR = f"{ROOT_DIR}/{args.data}/response"

splits = []
remaining_triplets = []

all_triplets = []
with open(f"{RESPONSE_DIR}/triplets.txt", "r") as fin:
    lines = fin.readlines()
    for line in lines:
        line = line.rstrip()
        all_triplets.append(line.split(" "))

graph_indices = UndirectedGraph(len(all_triplets))
for tr1_idx in range(0, len(all_triplets)):
    for tr2_idx in range(tr1_idx + 1, len(all_triplets) - 1):
        triplet1 = all_triplets[tr1_idx]
        triplet2 = all_triplets[tr2_idx]
        if set(triplet1) & set(triplet2) != set():
            graph_indices.add_edge(tr1_idx, tr2_idx)

with open(f"{RESPONSE_DIR}/triplets.txt", "r") as fin:
    triplet_lines = fin.readlines()
with open(f"{RESPONSE_DIR}/tripletAnswer.txt", "r") as fin:
    answer_lines = fin.readlines()
with open(f"{RESPONSE_DIR}/tripletDistribution.txt", "r") as fin:
    distribution_lines = fin.readlines()


def make_splits():
    global splits, remaining_triplets
    splits = [[] for i in range(10)]
    remaining_triplets = all_triplets.copy()
    while len(remaining_triplets) > 0:
        lengths = [len(s) for s in splits]
        split_id = np.argmin(lengths)
        current_triplets = splits[split_id]

        i = np.random.randint(0, len(remaining_triplets))
        current_triplet = remaining_triplets[i]
        current_triplets.append(current_triplet)
        del remaining_triplets[i]
        idx_i = all_triplets.index(current_triplet)
        for adj in graph_indices.adj[idx_i]:
            current_triplet = all_triplets[adj]
            current_triplets.append(current_triplet)
            if current_triplet in remaining_triplets:
                i = remaining_triplets.index(current_triplet)
                del remaining_triplets[i]

    print(f"remaining: {len(remaining_triplets)}")
    lengths = [len(s) for s in splits]
    print(f"lengths: {lengths}")
    return splits, remaining_triplets


def write_splits():
    os.makedirs(f"{RESPONSE_DIR}/splits3", exist_ok=True)
    for split_id in range(len(splits)):
        queries_total = 0
        with open(f"{RESPONSE_DIR}/splits3/train_triplets_{split_id}.txt", "w") as ftrain:
            with open(f"{RESPONSE_DIR}/splits3/test_triplets_{split_id}.txt", "w") as ftest:
                for triplet, answer, distribution in zip(triplet_lines, answer_lines, distribution_lines):
                    triplet = triplet.rstrip().split(" ")
                    answer = answer.rstrip()
                    distribution = [int(d) for d in distribution.rstrip().split(" ")]
                    if answer not in ['1', '2']:
                        continue
                    if answer == '1' and distribution[0] <= 5:
                        continue
                    if answer == '2' and distribution[1] <= 5:
                        continue
                    queries_total += 1
                    if triplet not in splits[split_id] and triplet not in remaining_triplets:
                        if answer == '1':
                            ftrain.write(f"{triplet[0]} {triplet[1]} {triplet[2]}\n")
                        else:
                            ftrain.write(f"{triplet[0]} {triplet[2]} {triplet[1]}\n")
                    else:
                        if answer == '1':
                            ftest.write(f"{triplet[0]} {triplet[1]} {triplet[2]}\n")
                        else:
                            ftest.write(f"{triplet[0]} {triplet[2]} {triplet[1]}\n")
        print(f"queries total: {queries_total}")


make_splits()
write_splits()
