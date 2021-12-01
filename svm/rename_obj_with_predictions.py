import argparse
import json
import logging
import sys
from os.path import dirname, realpath, join
import os

STYLE_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(STYLE_DIR)
from common.utils import set_logger_file, parse_buildings_csv, STYLES


LOGGER = logging.getLogger(__file__)


def process_building():
    with open(obj_file_in, "r") as fin:
        lines = fin.readlines()
    with open(style_predictions_file, "r") as fin:
        predictions = json.load(fin)
    with open(obj_file_out, "w") as fout:
        for line in lines:
            if line.startswith("o ") \
                    and "__Unknown" in line \
                    and any(s in line.lower() for s in ['tower', 'dome', 'window', 'door', 'column']):
                o, name = line.rstrip().split(" ")
                style = [s for n, s in predictions.items() if name in n]
                assert len(style) <= 1
                if len(style) == 1:
                    style = style[0]
                    name = name.replace("Unknown", style)
                else:
                    LOGGER.info(f"no style found for {name}")
                fout.write(f"o {name}\n")
            else:
                fout.write(line)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_dir', type=str, required=True)
    parser.add_argument('--style_predictions_dir', type=str, required=True)
    parser.add_argument('--buildings_csv', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = os.path.join(args.logs_dir, os.path.basename(args.buildings_csv).replace('.csv', '.log'))
    LOGGER = set_logger_file(_log_file, LOGGER)

    style_prefix = "style" + os.path.basename(args.style_predictions_dir).replace("style_predictions", "")

    LOGGER.info(f"Starting running {os.path.realpath(__file__)}...")

    buildings = parse_buildings_csv(args.buildings_csv)
    LOGGER.info("buildings: {}".format(buildings))

    for building in buildings:
        obj_file_in = os.path.join(args.obj_dir, building, "{}.obj".format(building))
        obj_file_out = os.path.join(args.obj_dir, building, "{}_{}.obj".format(building, style_prefix))
        style_predictions_file = os.path.join(args.style_predictions_dir, f"{building}.json")
        if not os.path.exists(obj_file_in):
            LOGGER.info("{} doesn't exist. Won't process.".format(obj_file_in))
            continue
        if not os.path.exists(style_predictions_file):
            LOGGER.info("{} doesn't exist. Won't process.".format(style_predictions_file))
            continue
        LOGGER.info("Processing {}".format(building))
        process_building()

    LOGGER.info(f"End running {os.path.realpath(__file__)}")
