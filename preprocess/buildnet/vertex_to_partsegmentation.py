import os
import json
import shutil
import argparse
from tqdm import tqdm
import pandas as pd
import logging


LOGGER = logging.getLogger(name="vertex_to_partsegmentation")

LABELS_TO_REMOVE = json.load(open("../resources/BUILDNET_ELEMENTS_TO_REMOVE.json"))
LABEL_MEANING = json.load(open("../resources/BUILDNET_ELEMENTS.json"))
STYLISTIC_PARTS = json.load(open("../resources/BUILDNET_STYLISTIC_ELEMENTS.json"))


VERTEX_LABEL = {}
O_ANNOTATIONS = {}
component_cnt = -1


def set_logger_file(log_file):
    global LOGGER
    file_handler = logging.FileHandler(log_file, 'a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    for hdlr in LOGGER.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr, logging.FileHandler):
            LOGGER.removeHandler(hdlr)
    LOGGER.addHandler(file_handler)
    LOGGER.setLevel(logging.INFO)


def reset():
    global VERTEX_LABEL, O_ANNOTATIONS, o_name, component_cnt
    VERTEX_LABEL = {}
    O_ANNOTATIONS = {}
    o_name = None
    component_cnt = -1


def process_files_line_by_line(file_in, file_out, process_line_fn):
    with open(file_in, "r") as f_in:
        with open(file_out, "w") as f_out:
            line = f_in.readline()
            cnt = 1
            process_line_fn(line, f_out)
            while line:
                line = f_in.readline()
                cnt += 1
                process_line_fn(line, f_out)


def process_annotation_file(filepath):
    global O_ANNOTATIONS
    with open(filepath, "r") as f:
        O_ANNOTATIONS = json.load(f)


def process_obj_line(line, f_out):
    global O_ANNOTATIONS, LABEL_MEANING, STYLISTIC_PARTS, component_cnt
    if line != "":
        line_split = line.strip().split(" ")
        if line_split[0] == "o":
            component_cnt += 1
            if len(line_split) > 1:
                o_name = line_split[1]
                o_annotation = O_ANNOTATIONS[o_name]
                label_meaning = LABEL_MEANING[str(o_annotation)]
                if label_meaning in STYLISTIC_PARTS:
                    changed_line = line.replace(line_split[1], "{}_{}_{}__unknown".format(component_cnt, o_name, label_meaning))
                    f_out.write(changed_line)
                elif label_meaning in LABELS_TO_REMOVE:
                    changed_line = line.replace(line_split[1], "{}_{}_{}_remove".format(component_cnt, o_name, label_meaning))
                    f_out.write(changed_line)
                else:
                    f_out.write(line)
            else:
                changed_line = line.replace(line_split[0], "{} {}_unlabeled".format(line_split[0], component_cnt))
                f_out.write(changed_line)
        else:
            f_out.write(line)
    else:
        f_out.write(line)


def process_obj_file(file_in, file_out):
    process_files_line_by_line(file_in, file_out, process_obj_line)


def process_one(textures_dir, labels_dir, obj_dir_in, obj_dir_out, building):
    reset()
    label_f = os.path.join(labels_dir, "{}_label.json".format(building))
    obj_name = "{}.obj".format(building)
    mtl_name = "{}.mtl".format(building)

    dir_out = os.path.join(obj_dir_out, building)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    try:
        shutil.copytree(os.path.join(textures_dir, building), os.path.join(dir_out, building))
    except Exception as e:
        LOGGER.info("Building {} is being skipped".format(building))
        return

    obj_f_in = os.path.join(obj_dir_in, obj_name)
    obj_f_out = os.path.join(dir_out, obj_name)
    mtl_f_in = os.path.join(obj_dir_in, mtl_name)
    mtl_f_out = os.path.join(dir_out, mtl_name)
    shutil.copy(mtl_f_in, mtl_f_out)
    process_annotation_file(label_f)
    process_obj_file(obj_f_in, obj_f_out)


def run(textures_dir, labels_dir, obj_dir_in, obj_dir_out, buildings_csv):

    if not os.path.exists(obj_dir_out):
        os.makedirs(obj_dir_out)

    buildings = pd.read_csv(buildings_csv, sep=";", header=None, names=["Style", 'Building'])
    buildings = buildings['Building'].values

    f_cnt = 0
    for building in tqdm(buildings):
        if f_cnt % 100 == 0:
            LOGGER.info("Processing {}".format(building))
        f_cnt += 1
        process_one(textures_dir, labels_dir, obj_dir_in, obj_dir_out, building)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/media/christina/Data/ANNFASS_data/BUILDNET_Buildings", type=str)
    parser.add_argument("--textures", default="raw_data/textures", type=str)
    parser.add_argument("--labels", default="raw_data/component_to_labels/GNN/label_32", type=str)
    parser.add_argument("--obj_in", default="raw_data/pycollada_unit_obj", type=str)
    parser.add_argument("--buildings", default="buildings.csv", type=str)
    parser.add_argument("--obj_out", default="normalizedObjBatch1", type=str)
    parser.add_argument("--logs_dir", default="/media/christina/Data/ANNFASS_data/BUILDNET_Buildings/logs", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    set_logger_file(os.path.join(args.logs_dir, os.path.basename(args.buildings).replace('.csv', '.log')))

    LOGGER.info("Textures: {}".format(os.path.join(args.root, args.textures)))
    LOGGER.info("Labels: {}".format(os.path.join(args.root, args.labels)))
    LOGGER.info("Obj in: {}".format(os.path.join(args.root, args.obj_in)))
    LOGGER.info("Obj out: {}".format(os.path.join(args.root, args.obj_out)))
    LOGGER.info("Buildings csv: {}".format(os.path.join(args.root, args.buildings)))

    run(os.path.join(args.root, args.textures),
        os.path.join(args.root, args.labels),
        os.path.join(args.root, args.obj_in),
        os.path.join(args.root, args.obj_out),
        os.path.join(args.root, args.buildings))