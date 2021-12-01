import argparse
import logging
import os
import sys
from os.path import dirname, realpath, join, basename

import cv2
import matplotlib.pyplot as plt

SOURCE_DIR = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(SOURCE_DIR)
from common.utils import parse_buildings_with_style_csv, set_logger_file
from common.matplotlib_utils import image_grid, figure_to_img
from common.utils import STYLES, BUILDNET_STYLISTIC_ELEMENTS


LOGGER = logging.getLogger(__name__)


def get_renderings_from_dir(directory):
    images = []
    name = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                img = cv2.imread(join(root, file))
                images.append(img)
                el = [el for el in BUILDNET_STYLISTIC_ELEMENTS if el.lower() in file.lower()][0]
                st = [s for s in STYLES if s.lower() in file.lower()][0]
                name = f"{el}__{st}"
    return images, name


def get_component_renderings(component_dir):
    images, component = get_renderings_from_dir(component_dir)
    title = f"{building}_{component}"
    len_images = len(images)
    if len_images % 3 == 0:
        fig = image_grid(images, title=title, rows=3, cols=len_images//3)
    elif len_images % 6 == 0:
        fig = image_grid(images, title=title, rows=6, cols=len_images//6)
    elif len_images % 5 == 0:
        fig = image_grid(images, title=title, rows=5, cols=len_images//5)
    elif len_images % 7 == 0:
        fig = image_grid(images, title=title, rows=7, cols=len_images//7)
    elif len_images % 4 == 0:
        fig = image_grid(images, title=title, rows=4, cols=len_images//4)
    else:
        cols = len_images // 2
        fig = image_grid(images[:2*cols], title=title, rows=2, cols=cols)

    img = figure_to_img(fig)
    return img, component


def run_building():
    for root, dirs, files in os.walk(join(args.ROOT_DIR, args.RENDERINGS_DIR)):
        for group_dir in dirs:
            if group_dir.startswith("group"):
                img, component = get_component_renderings(join(root, group_dir))
                el = [el for el in BUILDNET_STYLISTIC_ELEMENTS if el.lower() in component.lower()][0]
                img_file = join(args.ROOT_DIR, args.OUT_DIR, el, f"{building}_{group_dir}_{component}.png")
                os.makedirs(join(args.ROOT_DIR, args.OUT_DIR, el), exist_ok=True)
                cv2.imwrite(img_file, img)
                plt.close('all')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ROOT_DIR', type=str, required=True)
    parser.add_argument('-RENDERINGS_DIR', type=str, default="groups_june17_renderings/materials_on_daylight")
    parser.add_argument('-OUT_DIR', type=str, default="groups_june17_renderings/materials_on_daylight_plotted")
    parser.add_argument('-BUILDINGS_CSV', type=str, required=True)
    parser.add_argument('-LOGS_DIR', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.LOGS_DIR):
        os.makedirs(args.LOGS_DIR, exist_ok=True)
    log_file = os.path.join(args.LOGS_DIR, f'{basename(__file__)}.log')
    print("logs in ", log_file)
    set_logger_file(log_file, LOGGER)

    buildings = parse_buildings_with_style_csv(args.BUILDINGS_CSV)

    os.makedirs(join(args.ROOT_DIR, args.OUT_DIR), exist_ok=True)

    for building_style, building in buildings:
        run_building()
