import argparse
import json
import logging
import os
import sys

import matplotlib.image as mpimg
import numpy as np

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(SOURCE_DIR)
from common.utils import parse_buildings_csv, set_logger_file


LOGGER = logging.getLogger(__name__)
img_ext = "png"


def pct_reddish_pixels(rgb_image):
    red_values = rgb_image[:, :, 0]
    green_values = rgb_image[:, :, 1]
    blue_values = rgb_image[:, :, 2]
    reddish_pixels = np.sum(np.logical_and(red_values > 1.25 * green_values, red_values > 1.25 * blue_values))
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    pct = round((reddish_pixels/total_pixels)*100, 3)
    # print("the percentage of reddish pixels=", pct, "%")
    return pct


def view_must_be_discarded(filename):
    rgb_img = mpimg.imread(filename)
    # img = cv2.imread(filename)
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    distance_factor = int(os.path.basename(filename).split("_")[-2])
    pct_red = pct_reddish_pixels(rgb_img)

    if pct_red <= 2 and distance_factor < 2:
        return True, pct_red, distance_factor
    if pct_red <= 1 and distance_factor >= 2:
        return True, pct_red, distance_factor
    return False, pct_red, distance_factor


def rename(filename):
    # print("Renaming {}".format(filename))
    os.rename(filename, filename.replace(f".{img_ext}", f"_discard.{img_ext}"))


def undo_rename(filename):
    os.rename(filename, filename.replace(f"_discard.{img_ext}", f".{img_ext}"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--viewpoints_dir", default="viewpoints", type=str)
    parser.add_argument("--buildings_csv", required=True, type=str)
    parser.add_argument("--logs_dir", required=True, type=str)
    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    log_file = os.path.join(args.logs_dir, f"{os.path.basename(os.path.realpath(__file__))}.log")
    print("logs in ", log_file)
    set_logger_file(log_file, LOGGER)

    LOGGER.info("Un-discarding viewpoints..")
    for root, dirs, files in os.walk(args.viewpoints_dir):
        for file in files:
            if f"_discard.{img_ext}" in file:
                undo_rename(os.path.join(root, file))

    LOGGER.info("Selecting viewpoints ...")
    LOGGER.info("viewpoints_dir: {}".format(args.viewpoints_dir))
    LOGGER.info("buildings_csv: {}".format(args.buildings_csv))
    LOGGER.info("logs_dir: {}".format(args.logs_dir))

    buildings = parse_buildings_csv(args.buildings_csv)
    LOGGER.info("buildings = {}".format(buildings))

    for building in buildings:
        viewpoints_dir = os.path.join(args.viewpoints_dir, building)
        if not os.path.exists(viewpoints_dir):
            LOGGER.info("Skipping {} as viewpoints dir doesnt exist".format(building))
            continue
        LOGGER.info("Processing {}".format(building))

        for root, dirs, files in os.walk(viewpoints_dir):
            for file in files:
                if ".json" in file and not "selected" in file:
                    group = ''.join(filter(str.isdigit, file))
                    per_dist_scores = {}
                    with open(os.path.join(root, file), "r") as fin:
                        view_points = json.load(fin)
                    for viewpoint in view_points:
                        obj_name, face_idx, dist_idx, face_center, direction, camera_location, angle = viewpoint
                        jpg_file = "{}_{}_{}_{}.{}".format(obj_name, face_idx, dist_idx, angle, img_ext)
                        jpg_file = os.path.join(root, "group_{}".format(group), jpg_file)

                        if not os.path.exists(jpg_file):
                            LOGGER.warning("File {} doesnt exist. Go to next group.".format(jpg_file))
                            break

                        mustdiscard, pctred, distf = view_must_be_discarded(jpg_file)
                        if not mustdiscard:
                            per_dist_scores.setdefault(distf, [])
                            per_dist_scores[distf].append((jpg_file, viewpoint, pctred))
                        else:
                            rename(jpg_file)

                    selected_view_points = []
                    for distf in per_dist_scores:
                        res = sorted(per_dist_scores[distf], key=lambda x: x[2], reverse=True)
                        for (f, v, s) in res[0:6]:
                            selected_view_points.append(v)
                        for (f, v, s) in res[6:]:
                            rename(f)

                    with open(os.path.join(root, file.replace(".json", "_selected.json")), "w") as fout:
                        json.dump(selected_view_points, fout)
