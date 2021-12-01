import argparse
import logging
import os
import sys
from os.path import dirname, realpath, join, basename, exists

import numpy as np
import pandas as pd

SOURCE_DIR = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(SOURCE_DIR)
from common.utils import parse_buildings_csv, set_logger_file, BUILDNET_STYLISTIC_ELEMENTS


LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dir', type=str)
    parser.add_argument('-unique_dir', type=str, default="unique_point_clouds")
    parser.add_argument('-out_dir', type=str, default="stats_with_images")
    parser.add_argument('-buildings_csv', type=str)
    parser.add_argument('-logs_dir', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)
    log_file = os.path.join(args.logs_dir, f'{basename(__file__)}.log')
    print("logs in ", log_file)
    set_logger_file(log_file, LOGGER)

    buildings = parse_buildings_csv(args.buildings_csv)

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    stats = np.zeros((len(BUILDNET_STYLISTIC_ELEMENTS), len(buildings)))

    for col, building in enumerate(buildings):
        if not exists(join(args.root_dir, args.unique_dir, building)):
            continue
        unique_components = os.listdir(join(args.root_dir, args.unique_dir, building))
        for row, semantic in enumerate(BUILDNET_STYLISTIC_ELEMENTS):
            unique_semantic = [f for f in unique_components if f.endswith(".ply") and semantic.lower() in f.lower()]
            if len(unique_semantic) > 0:
                stats[row][col] = len(unique_components)

    df = pd.DataFrame(stats, index=BUILDNET_STYLISTIC_ELEMENTS, columns=buildings)
    total = df.sum(axis=1)
    occurrences = df.astype(bool).sum(axis=1)
    avg = total / occurrences
    median = df.replace(0, np.NaN).median(axis=1)
    df['total occurrences'] = total
    df['number of buildings it occurs'] = occurrences
    df['avg occurrence per building'] = avg
    df['median occurrence per building'] = median
    df['semantic label'] = BUILDNET_STYLISTIC_ELEMENTS
    stats_file = join(args.root_dir, "building_semantic_stats.csv")
    df.to_csv(stats_file)

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    stats = {}
    for building in buildings:
        if not exists(join(args.root_dir, args.unique_dir, building)):
            continue
        unique_components = os.listdir(join(args.root_dir, args.unique_dir, building))
        building_type = building.split("_mesh")[0]
        for semantic in BUILDNET_STYLISTIC_ELEMENTS:
            unique_semantic = [f for f in unique_components if f.endswith(".ply") and semantic.lower() in f.lower()]
            if len(unique_semantic) > 0:
                stats.setdefault(semantic, {})
                stats[semantic].setdefault(building_type, 0)
                stats[semantic][building_type] += len(unique_components)

    df = pd.DataFrame(stats)
    stats_file = join(args.root_dir, "buildingtype_semantic_stats.csv")
    df.to_csv(stats_file)
