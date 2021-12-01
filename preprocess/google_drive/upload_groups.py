from __future__ import print_function

import argparse
import logging
import sys
from os.path import dirname

from googleapiclient.discovery import build

from gd_utils import *

STYLES_DIR = dirname(dirname(dirname(os.path.realpath(__file__))))
sys.path.append(STYLES_DIR)
from common.utils import parse_buildings_csv, set_logger_file

LOGGER = logging.getLogger(name="upload")


def upload_data(root_id, groups_path, buildings_csv):

    buildings = parse_buildings_csv(buildings_csv)

    groups_dir_id = make_folder_if_not_exists(service, "Groups", LOGGER, root_id)

    for building in buildings:
        groups_dir = os.path.join(groups_path, building)
        if not os.path.exists(groups_dir):
            continue
        LOGGER.info("Uploading for {}".format(building))
        for f in os.listdir(groups_dir):
            if "_grouped_" in f:
                use_name = "{}_{}".format(building, f)
                filepath = os.path.join(groups_dir, f)
                update_or_create_image_in_existing_dir(service, use_name, groups_dir_id, filepath, LOGGER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--gdrepo", default="BUILDNET_Buildings_groups_june", type=str)
    parser.add_argument("--groups_path", default="groups", type=str)
    parser.add_argument("--buildings_csv", required=True, type=str)
    parser.add_argument("--logs_dir", default="upload.log", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = os.path.join(args.logs_dir, os.path.basename(args.buildings_csv).replace('.csv', '.log'))
    set_logger_file(_log_file, LOGGER)

    LOGGER.info("root_dir: {}".format(args.root_dir))
    LOGGER.info("gdrepo: {}".format(args.gdrepo))
    LOGGER.info("groups_path: {}".format(args.groups_path))
    LOGGER.info("buildings_csv: {}".format(args.buildings_csv))

    service = build('drive', 'v3', credentials=get_and_save_access())

    root_dir_id = make_folder_if_not_exists(service, args.gdrepo, LOGGER, parent_id=get_folder_id(service, 'ANNFASS'))

    upload_data(root_dir_id,
                os.path.join(args.root_dir, args.groups_path),
                args.buildings_csv)

