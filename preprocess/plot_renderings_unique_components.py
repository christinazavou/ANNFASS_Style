import argparse
import json
import logging
import os
import sys
from os.path import dirname, realpath, join, basename, exists

import cv2
import matplotlib.pyplot as plt

SOURCE_DIR = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(SOURCE_DIR)
from common.utils import parse_buildings_with_style_csv, set_logger_file
from common.matplotlib_utils import image_grid, render_views, figure_to_img


LOGGER = logging.getLogger(__name__)


def get_renderings_from_dir(directory, prefix):
    img0 = cv2.imread(join(directory, f"{prefix}r0.png"))
    img1 = cv2.imread(join(directory, f"{prefix}r1.png"))
    img2 = cv2.imread(join(directory, f"{prefix}r2.png"))
    img3 = cv2.imread(join(directory, f"{prefix}r3.png"))
    img4 = cv2.imread(join(directory, f"{prefix}r4.png"))
    img5 = cv2.imread(join(directory, f"{prefix}.png"))
    return [img0, img1, img2, img3, img4, img5]


def get_component_renderings(component_dir):
    # images = get_renderings_from_dir(component_dir, "img")
    images = get_renderings_from_dir(component_dir, "tr_img")
    title = basename(component_dir).replace("style_mesh_", "")
    group = title.split("_")[0]
    title = title.replace(f"{group}_", f"{group}\n")
    style = title.split("_")[-1]
    # title = title.replace(f"_{style}", f"\n{style}")
    title = title.replace(f"_{style}", f"\n{building_style}")
    img = render_views(images, title=title, color='white', as_img=True)
    return img


def run_building():
    # try:
    grouping_images = get_renderings_from_dir(join(args.ROOT_DIR, args.GROUPS_DIR, building), "tr_group_")
    # except:
    #     grouping_images = get_renderings_from_dir(join(args.ROOT_DIR, args.GROUPS_DIR, building), "_grouped_")
    grouping_images = [grouping_images[0], grouping_images[2], grouping_images[4]]
    grouping_img = render_views(grouping_images, as_img=True)

    component_images = []
    for component in unique_components:
        img = get_component_renderings(join(args.ROOT_DIR, args.COMPONENTS_DIR, building, component))
        component_images.append(img)
    if len(component_images) == 0:
        return
    i = 0
    for idx in range(0, len(component_images), 5):
        i += 1
        img_file = join(args.ROOT_DIR, args.OUT_DIR, f"{building}_components{i}.png")
        current_images = component_images[idx:idx+5]
        current_images.append(grouping_img)
        fig = image_grid(current_images, title=building, cols=len(current_images), rows=1)
        img = figure_to_img(fig)
        cv2.imwrite(img_file, img)
    plt.close('all')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-RENDER_MODE', type=int, default=1, help="1: freestyle, 2: materials")
    parser.add_argument('-ROOT_DIR', type=str)
    parser.add_argument('-GROUPS_DIR', type=str, default="groups_june17")
    parser.add_argument('-COMPONENTS_DIR', type=str, default="unified_normalized_components")
    parser.add_argument('-UNIQUE_DIR', type=str, default="unique_point_clouds")
    parser.add_argument('-OUT_DIR', type=str, default="plotted_renderings")
    parser.add_argument('-BUILDINGS_CSV', type=str)
    parser.add_argument('-LOGS_DIR', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.LOGS_DIR):
        os.makedirs(args.LOGS_DIR, exist_ok=True)
    log_file = os.path.join(args.LOGS_DIR, f'{basename(__file__)}.log')
    print("logs in ", log_file)
    set_logger_file(log_file, LOGGER)

    buildings = parse_buildings_with_style_csv(args.BUILDINGS_CSV)

    os.makedirs(join(args.ROOT_DIR, args.OUT_DIR), exist_ok=True)

    for building_style, building in buildings:
        unique_file = join(args.ROOT_DIR, args.UNIQUE_DIR, building, "duplicates.json")
        if not exists(unique_file):
            continue
        with open(unique_file, "r") as fin:
            unique_components = json.load(fin).keys()
            LOGGER.info(unique_components)
        run_building()
