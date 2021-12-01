import argparse
import os
import sys

import numpy as np
import pandas as pd

from utils import read_obj_with_components, read_face_indices, fix_component

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.utils import STYLES


def store_encodings_and_labels(encodings_dict, output_dir, stats_file):
    building_cnts = dict()
    for ci, component in enumerate(encodings_dict):
        num_comp_points = len(encodings_dict[component])
        building_cnts[component] = num_comp_points
        if num_comp_points > 0:
            encodings = np.stack(encodings_dict[component])
            labels = np.zeros((1, len(STYLES))).astype(np.int32)
            for idx, _style in enumerate(STYLES):
                if _style.lower() in component.lower():
                    labels[0, idx] = 1
            os.makedirs(os.path.join(output_dir, 'sum', building), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'max', building), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'avg', building), exist_ok=True)
            with open(os.path.join(output_dir, 'sum', building, "group{}_{}.npy".format(ci, component)), 'wb') as f:
                np.save(f, np.sum(encodings, 0))
            with open(os.path.join(output_dir, 'sum', building, "group{}_{}_labels.npy".format(ci, component)), 'wb') as f:
                np.save(f, labels)
            with open(os.path.join(output_dir, 'max', building, "group{}_{}.npy".format(ci, component)), 'wb') as f:
                np.save(f, np.max(encodings, 0))
            with open(os.path.join(output_dir, 'max', building, "group{}_{}_labels.npy".format(ci, component)), 'wb') as f:
                np.save(f, labels)
            with open(os.path.join(output_dir, 'avg', building, "group{}_{}.npy".format(ci, component)), 'wb') as f:
                np.save(f, np.mean(encodings, 0))
            with open(os.path.join(output_dir, 'avg', building, "group{}_{}_labels.npy".format(ci, component)), 'wb') as f:
                np.save(f, labels)
    if len(building_cnts) > 0:
        values = np.array(list(building_cnts.values()))
        building_cnts['minimum_cnt'] = np.min(values)
        building_cnts['maximum_cnt'] = np.max(values)
        building_cnts['avg_cnt'] = np.mean(values)
        building_cnts['total'] = len(values)
        pd.DataFrame.from_dict(building_cnts, orient='index').to_csv(stats_file)


def store_building_samples(component_references, components, face_indices, encodings_per_point):
    encodings_dict = {}
    for point_idx, face_idx in enumerate(face_indices):
        component = components[component_references[face_idx]][0, 0]
        component = fix_component(component)
        if any(s.lower() in component.lower() for s in STYLES):
            if component in encodings_dict:
                encodings_dict[component].append(encodings_per_point[point_idx])
            else:
                encodings_dict[component] = [encodings_per_point[point_idx]]
    return encodings_dict


def store_building_rnv_samples(component_references, components, face_indices, rnv_values, encodings_per_point):
    encodings_dict = {}
    for point_idx, face_idx in enumerate(face_indices):
        rnv_value = rnv_values[point_idx]
        if rnv_value == 0:
            continue  # the point is neither ridge nor valley
        component = components[component_references[face_idx]][0, 0]
        component = fix_component(component)
        if any(s.lower() in component.lower() for s in STYLES):
            if component in encodings_dict:
                encodings_dict[component].append(encodings_per_point[point_idx])
            else:
                encodings_dict[component] = [encodings_per_point[point_idx]]
    return encodings_dict


def run():
    encodings_per_point = np.load(features_file)
    total_points = encodings_per_point.shape[0]
    print("encodings loaded: {}".format(encodings_per_point.shape))

    try:
        enc = store_building_samples(component_references, components, face_indices[:total_points],
                                     encodings_per_point)
        store_encodings_and_labels(enc, out_dir, building_stats_csv)

        if use_rnv:
            enc = store_building_rnv_samples(component_references, components, face_indices[:total_points],
                                             rnvs[:total_points], encodings_per_point)
            store_encodings_and_labels(enc, out_dir_rnv, building_stats_rnv_csv)
    except Exception as e:
        if "Couldn't store_building_samples for component" in str(e):
            print("{} thus skipping building ...".format(e))
        else:
            raise e


def store_encoding_and_label(encodings, output_dir):
    if len(encodings) > 0:
        labels = np.zeros((1, len(STYLES))).astype(np.int32)
        for idx, _style in enumerate(STYLES):
            if _style.lower() in component.lower():
                labels[0, idx] = 1
        ci = list(building_cnts_c.keys()).index(component)
        os.makedirs(os.path.join(output_dir, 'sum', building), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'max', building), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'avg', building), exist_ok=True)
        with open(os.path.join(output_dir, 'sum', building, "group{}_{}.npy".format(ci, component)), 'wb') as f:
            np.save(f, np.sum(encodings, 0))
        with open(os.path.join(output_dir, 'sum', building, "group{}_{}_labels.npy".format(ci, component)), 'wb') as f:
            np.save(f, labels)
        with open(os.path.join(output_dir, 'max', building, "group{}_{}.npy".format(ci, component)), 'wb') as f:
            np.save(f, np.max(encodings, 0))
        with open(os.path.join(output_dir, 'max', building, "group{}_{}_labels.npy".format(ci, component)), 'wb') as f:
            np.save(f, labels)
        with open(os.path.join(output_dir, 'avg', building, "group{}_{}.npy".format(ci, component)), 'wb') as f:
            np.save(f, np.mean(encodings, 0))
        with open(os.path.join(output_dir, 'avg', building, "group{}_{}_labels.npy".format(ci, component)), 'wb') as f:
            np.save(f, labels)


def run_component():
    encodings_per_point = np.load(feature_file)
    print("encodings loaded: {}".format(encodings_per_point.shape))
    building_cnts_c[component] = len(encodings_per_point)

    if use_rnv:
        rnv_values = rnvs[:len(encodings_per_point)]
    indices = []
    for point_idx in range(len(encodings_per_point)):
        if use_rnv:
            rnv_value = rnv_values[point_idx]
            if rnv_value == 0:
                continue  # the point is neither ridge nor valley
        indices.append(point_idx)
    building_cnts_c_rnv[component] = len(indices)

    try:
        store_encoding_and_label(encodings_per_point, out_dir)
        if use_rnv:
            store_encoding_and_label(encodings_per_point[indices], out_dir_rnv)
    except Exception as e:
        if "Couldn't store_building_samples for component" in str(e):
            print("{} thus skipping building ...".format(e))
        else:
            raise e


def parse_buildings_csv(filename):
    buildings = []
    with open(filename, "r") as f:
        for line in f:
            buildings.append(line.strip().split(";")[1])
    print("buildings to process: {}".format(buildings))
    return buildings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/media/christina/Data/ANNFASS_data", type=str)
    parser.add_argument("--features_dir", required=True, type=str)
    parser.add_argument("--face_indices_dir", required=True, type=str)
    parser.add_argument("--rnv_dir", required=True, type=str)
    parser.add_argument("--normalizedObj_dir", default="ANNFASS_Buildings/normalizedObj", type=str)
    parser.add_argument("--buildings_csv", required=True, type=str)
    args = parser.parse_args()
    print(args)

    LAYERS = ['layer_n-1_features', 'layer_n-2_features', 'layer_n-3_features',
              'feature_concat', 'feature_bilinear', 'convd0_nearinterp']
    METHODS = ['max', 'weighted_sum', 'as_is']

    features_dir = args.features_dir
    face_indices_dir = os.path.join(args.root_dir, args.face_indices_dir)
    normalizedObj_dir = os.path.join(args.root_dir, args.normalizedObj_dir)
    rnv_dir = os.path.join(args.root_dir, args.rnv_dir)

    buildings = parse_buildings_csv(args.buildings_csv)
    for building in buildings:
        building_cnts_c = dict()
        building_cnts_c_rnv = dict()
        obj_file = os.path.join(normalizedObj_dir, building, "{}.obj".format(building))
        if not os.path.exists(obj_file):
            print("no obj file {}".format(obj_file))
            continue
        face_indices_file = os.path.join(face_indices_dir, "{}.txt".format(building))
        if not os.path.exists(face_indices_file):
            print("no indices file {}".format(face_indices_file))
            continue
        if os.path.exists(rnv_dir):
            rnv_file = os.path.join(rnv_dir, "{}.txt".format(building))
            use_rnv = True
        else:
            use_rnv = False
            print("WARNING: rnvs wont be used")

        print("run for {}".format(building))
        face_indices = read_face_indices(face_indices_file)
        obj_with_components = read_obj_with_components(obj_file)
        vertices, faces, component_references, components = obj_with_components
        assert np.max(face_indices) <= faces.shape[0] - 1
        if use_rnv:
            rnvs = read_face_indices(rnv_file)

        for layer_dir in LAYERS:
            for method_dir in METHODS:
                if os.path.exists(os.path.join(features_dir, layer_dir, method_dir)):
                    features_file = os.path.join(features_dir, layer_dir, method_dir, "{}.npy".format(building))
                    if os.path.isfile(features_file):
                        building_stats_csv = os.path.join(features_dir, "{}_stats.csv".format(building))
                        building_stats_rnv_csv = os.path.join(features_dir, "{}_stats_rnv.csv".format(building))
                        out_dir = os.path.join(features_dir, layer_dir, method_dir + "_per_component")
                        out_dir_rnv = os.path.join(features_dir, layer_dir, method_dir + "_per_component_rnv")
                        run()
                    else:
                        for component in components:
                            component = component[0]
                            if "__" not in component:
                                continue
                            feature_file = os.path.join(features_dir, layer_dir, method_dir, "{}_style_mesh_{}.npy".format(building, component))
                            if os.path.isfile(feature_file):
                                out_dir = os.path.join(features_dir, layer_dir, method_dir + "_per_component")
                                out_dir_rnv = os.path.join(features_dir, layer_dir, method_dir + "_per_component_rnv")
                                run_component()
                            else:
                                print("{} doesnt exist".format(feature_file))

        if building_cnts_c != {}:
            building_stats_csv = os.path.join(features_dir, "{}_stats.csv".format(building))
            values = np.array(list(building_cnts_c.values()))
            building_cnts_c['minimum_cnt'] = np.min(values)
            building_cnts_c['maximum_cnt'] = np.max(values)
            building_cnts_c['avg_cnt'] = np.mean(values)
            building_cnts_c['total'] = len(values)
            pd.DataFrame.from_dict(building_cnts_c, orient='index').to_csv(building_stats_csv)
        if building_cnts_c_rnv != {}:
            building_stats_csv = os.path.join(features_dir, "{}_stats_rnv.csv".format(building))
            values = np.array(list(building_cnts_c_rnv.values()))
            building_cnts_c_rnv['minimum_cnt'] = np.min(values)
            building_cnts_c_rnv['maximum_cnt'] = np.max(values)
            building_cnts_c_rnv['avg_cnt'] = np.mean(values)
            building_cnts_c_rnv['total'] = len(values)
            pd.DataFrame.from_dict(building_cnts_c_rnv, orient='index').to_csv(building_stats_csv)

