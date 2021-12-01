import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from utils import read_obj_with_components, read_face_indices, fix_groups, \
    fix_component

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.utils import STYLES, parse_buildings_csv


def store_encodings_and_labels(encodings_dict, out_folder):
    building_cnts = dict()
    for component_group in encodings_dict:
        building_cnts.setdefault(component_group, 0)
        for component in encodings_dict[component_group]:
            building_cnts[component_group] += len(encodings_dict[component_group][component])
            encodings = np.stack(encodings_dict[component_group][component])
            labels = np.zeros((1, len(STYLES))).astype(np.int32)
            for idx, _style in enumerate(STYLES):
                if _style.lower() in component.lower():
                    labels[0, idx] = 1

            os.makedirs(os.path.join(out_folder, 'sum', building), exist_ok=True)
            os.makedirs(os.path.join(out_folder, 'max', building), exist_ok=True)
            os.makedirs(os.path.join(out_folder, 'avg', building), exist_ok=True)
            with open(os.path.join(out_folder, 'sum', building, "group{}_{}.npy".format(component_group, component)), 'wb') as f:
                np.save(f, np.sum(encodings, 0))
            with open(os.path.join(out_folder, 'sum', building, "group{}_{}_labels.npy".format(component_group, component)), 'wb') as f:
                np.save(f, labels)
            with open(os.path.join(out_folder, 'max', building, "group{}_{}.npy".format(component_group, component)), 'wb') as f:
                np.save(f, np.max(encodings, 0))
            with open(os.path.join(out_folder, 'max', building, "group{}_{}_labels.npy".format(component_group, component)), 'wb') as f:
                np.save(f, labels)
            with open(os.path.join(out_folder, 'avg', building, "group{}_{}.npy".format(component_group, component)), 'wb') as f:
                np.save(f, np.mean(encodings, 0))
            with open(os.path.join(out_folder, 'avg', building, "group{}_{}_labels.npy".format(component_group, component)), 'wb') as f:
                np.save(f, labels)

    if len(building_cnts) > 0:
        values = np.array(list(building_cnts.values()))
        building_cnts['minimum_cnt'] = np.min(values)
        building_cnts['maximum_cnt'] = np.max(values)
        building_cnts['avg_cnt'] = np.mean(values)
        building_cnts['total'] = len(values)
        pd.DataFrame.from_dict(building_cnts, orient='index').to_csv(building_stats_csv)


def get_component_group(component, groups):
    for group_name, group_components in groups.items():
        if component in group_components:
            return group_name
    raise Exception("Couldn't find group for component {}".format(component))


def store_building_samples(groups, component_references, components, face_indices, encodings_per_point):
    encodings_dict = {}
    for point_idx, face_idx in enumerate(face_indices):
        component = components[component_references[face_idx]][0, 0]
        component = fix_component(component)
        if any(s.lower() in component.lower() for s in STYLES):
            component_group = get_component_group(component, groups)

            if component_group not in encodings_dict:
                encodings_dict[component_group] = {}

            if component in encodings_dict[component_group]:
                encodings_dict[component_group][component].append(encodings_per_point[point_idx])
            else:
                encodings_dict[component_group][component] = [encodings_per_point[point_idx]]

    return encodings_dict


def run():
    os.makedirs(out_dir, exist_ok=True)
    print("run for {}".format(building))
    encodings_per_point = np.load(features_file)
    total_points = encodings_per_point.shape[0]
    print("encodings loaded: {}".format(encodings_per_point.shape))
    try:
        enc = store_building_samples(groups, component_references, components, face_indices[:total_points], encodings_per_point)
        store_encodings_and_labels(enc, out_dir)
        return enc
    except Exception as e:
        if "Couldn't find group for component" in str(e):
            print("{} thus skipping building ...".format(e))
        return None


if __name__ == '__main__':

    LAYERS = ['layer_n-1_features', 'layer_n-2_features', 'layer_n-3_features',
              'feature_concat', 'feature_bilinear', 'convd0_nearinterp']
    METHODS = ['max', 'weighted_sum', 'as_is']

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/media/christina/Data/ANNFASS_data", type=str)
    parser.add_argument("--features_dir", required=True, type=str)
    parser.add_argument("--face_indices_dir", required=True, type=str)
    parser.add_argument("--normalizedObj_dir", default="ANNFASS_Buildings/normalizedObj", type=str)
    parser.add_argument("--groups_dir", default="ANNFASS_Buildings/groups", type=str)
    parser.add_argument("--buildings_csv", default="/media/christina/Data/ANNFASS_data/ANNFASS_Buildings/buildings.csv", type=str)
    args = parser.parse_args()

    features_dir = args.features_dir
    face_indices_dir = os.path.join(args.root_dir, args.face_indices_dir)
    normalizedObj_dir = os.path.join(args.root_dir, args.normalizedObj_dir)
    groups_dir = os.path.join(args.root_dir, args.groups_dir)
    buildings_csv = args.buildings_csv

    buildings = parse_buildings_csv(buildings_csv)

    for building in buildings:

        groups_file = os.path.join(groups_dir, building, "groups.json")
        fixed_groups_file = os.path.join(groups_dir, building, "groups_fixed.json")
        face_indices_file = os.path.join(face_indices_dir, "{}.txt".format(building))
        obj_file = os.path.join(normalizedObj_dir, building, "{}.obj".format(building))
        building_stats_csv = os.path.join(features_dir, "{}_stats.csv".format(building))

        if not os.path.exists(groups_file):
            print("no groups file {}".format(groups_file))
            continue
        if not os.path.exists(obj_file):
            print("no obj file {}".format(obj_file))
            continue
        if not os.path.exists(face_indices_file):
            print("no indices file {}".format(face_indices_file))
            continue
        fix_groups(groups_file, fixed_groups_file)
        groups_file = fixed_groups_file

        face_indices = read_face_indices(face_indices_file)
        with open(groups_file, "r") as fin:
            groups = json.load(fin)
        obj_with_components = read_obj_with_components(obj_file)
        vertices, faces, component_references, components = obj_with_components
        assert np.max(face_indices) <= faces.shape[0] - 1

        for layer_dir in LAYERS:
            for method_dir in METHODS:
                if os.path.exists(os.path.join(features_dir, layer_dir, method_dir)):
                    features_file = os.path.join(features_dir, layer_dir, method_dir, "{}.npy".format(building))
                    if not os.path.exists(features_file):
                        print("no features file {}".format(features_file))
                        continue
                    out_dir = os.path.join(features_dir, layer_dir, method_dir + "_per_component")
                    found_groups = run()
                    if found_groups is not None:
                        missing_groups = [g for g in groups.keys() if g not in found_groups]
                        if len(missing_groups) >0:
                            print(f"missing groups: {missing_groups} for building {building}")

# note: Now i will take all sampled points that lie within my components ==> i might have components with no samples
