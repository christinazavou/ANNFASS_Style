import json
import os
import argparse
from random import shuffle

import pandas as pd

pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--buildings_csv", required=True, type=str)
    parser.add_argument("--split_root", required=True, type=str)
    args = parser.parse_args()

    df_init = pd.read_csv(args.buildings_csv, sep=";", names=['style', 'building'])
    df_init = df_init[df_init['style'] == 'Unknown Style']

    all_buildings = list(df_init['building'].values)
    indices = list(range(len(all_buildings)))
    shuffle(indices)

    train_size = int(len(indices) * 0.8)
    val_size = int(train_size * 0.2)
    train_indices = indices[0: train_size-val_size]
    val_indices = indices[train_size-val_size:train_size]
    test_indices = indices[train_size:]

    os.makedirs(args.split_root, exist_ok=True)

    split_dir = os.path.join(args.split_root, "setA_train_val_test")
    os.makedirs(split_dir, exist_ok=True)

    train_file = os.path.join(split_dir,  "train.txt")
    val_file = os.path.join(split_dir,  "val.txt")
    test_file = os.path.join(split_dir,  "test.txt")
    all_file = os.path.join(split_dir,  "all.txt")
    splits_file = os.path.join(split_dir,  "split.json")

    train_buildings = [b for i, b in enumerate(all_buildings) if i in train_indices]
    test_buildings = [b for i, b in enumerate(all_buildings) if i in test_indices]
    val_buildings = [b for i, b in enumerate(all_buildings) if i in val_indices]

    with open(train_file, "w") as f_out:
        for building in train_buildings:
            f_out.write(f"{building}\n")
    with open(test_file, "w") as f_out:
        for building in test_buildings:
            f_out.write(f"{building}\n")
    with open(val_file, "w") as f_out:
        for building in val_buildings:
            f_out.write(f"{building}\n")
    with open(all_file, "w") as f_out:
        for building in train_buildings+test_buildings+val_buildings:
            f_out.write(f"{building}\n")

    with open(splits_file, "w") as f_out:
        json.dump({'train_buildings':train_buildings, 'test_buildings': test_buildings, 'val_buildings': val_buildings},
                  f_out, indent=4)
